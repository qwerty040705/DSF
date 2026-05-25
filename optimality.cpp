#include "optimality.cuh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

#include "experimental/trimesh_dsf32d/algorithm.cuh"
#include "global/include/arithmetic/math.cuh"

namespace cuDSF {
namespace evaluation_metrics {
namespace {

constexpr char DEFAULT_DSF_CSV_PATH[] = "D:/DongBin/cuDSF/dsf_vertices_all.csv";
constexpr char DEFAULT_MESH_PATH[] = "D:/DongBin/cuDSF/bunny.obj";
constexpr char DEFAULT_OUTPUT_DIR[] = "D:/DongBin/cuDSF/evaluation_metrics/results";
constexpr bool USE_NEAREST_TRIANGLES = false;
[[maybe_unused]] constexpr bool SAVE_TRIANGLE_RAW = false;
constexpr bool DEBUG_SMALL_RUN = false;
constexpr int DEFAULT_NUM_SAMPLES = 1000;
constexpr int TRIANGLES_PER_SAMPLE = 1;
constexpr float TRANSLATION_RANGE = 20.0f;
constexpr double ETA = 1e-6;
constexpr std::uint32_t RNG_SEED = 20260522u;
constexpr int NUM_DSF_VERTICES = 32;

const std::array<const char*, 5> SHAPE_ORDER = {
    "round", "normal", "long", "very_long", "extreme_long"};
const std::array<ScalingMethod, 3> SCALING_METHODS = {
    ScalingMethod::Raw, ScalingMethod::Max1, ScalingMethod::MaxFmax};
const std::array<int, 6> P_VALUES = {4, 8, 16, 32, 64, 128};

struct Arguments {
  std::string mesh_path = DEFAULT_MESH_PATH;
  std::string out_dir = DEFAULT_OUTPUT_DIR;
  int num_samples = DEFAULT_NUM_SAMPLES;
  int triangles_per_sample = TRIANGLES_PER_SAMPLE;
};

std::string scaling_method_name(ScalingMethod method) {
  switch (method) {
    case ScalingMethod::Raw:
      return "Raw";
    case ScalingMethod::Max1:
      return "Max1";
    case ScalingMethod::MaxFmax:
      return "MaxFmax";
  }

  throw std::runtime_error("Unknown scaling method.");
}

bool approximately_equal(double lhs, double rhs, double tolerance = 1e-5) {
  const double scale = std::max(1.0, std::max(std::abs(lhs), std::abs(rhs)));
  return std::abs(lhs - rhs) <= tolerance * scale;
}

double finite_or_zero(double value) {
  return std::isfinite(value) ? value : 0.0;
}

std::vector<std::string> split_csv_line(const std::string& line) {
  std::vector<std::string> cells;
  std::stringstream stream(line);
  std::string cell;

  while (std::getline(stream, cell, ',')) {
    if (!cell.empty() && cell.back() == '\r') {
      cell.pop_back();
    }
    cells.push_back(cell);
  }

  return cells;
}

DSFVertexRow parse_dsf_row(const std::vector<std::string>& cells) {
  if (cells.size() != 12) {
    throw std::runtime_error("Unexpected DSF CSV column count.");
  }

  DSFVertexRow row;
  row.shape_type = cells[0];
  row.R = std::stod(cells[1]);
  row.log10_R = std::stod(cells[2]);
  row.index = std::stoi(cells[3]);
  row.r_i = std::stod(cells[4]);
  row.u_x = std::stod(cells[5]);
  row.u_y = std::stod(cells[6]);
  row.u_z = std::stod(cells[7]);
  row.v_x = std::stod(cells[8]);
  row.v_y = std::stod(cells[9]);
  row.v_z = std::stod(cells[10]);
  row.norm_v = std::stod(cells[11]);
  return row;
}

void compute_triangle_centroids(MeshData& mesh) {
  mesh.triangle_centroids.clear();
  mesh.triangle_centroids.reserve(mesh.triangles.size());

  for (const std::array<int, 3>& triangle : mesh.triangles) {
    const vector4f& a = mesh.vertices[triangle[0]];
    const vector4f& b = mesh.vertices[triangle[1]];
    const vector4f& c = mesh.vertices[triangle[2]];
    mesh.triangle_centroids.push_back((a + b + c) / 3.0f);
  }
}

int parse_obj_face_index(const std::string& token, int num_vertices) {
  const std::size_t slash = token.find('/');
  const std::string index_text = token.substr(0, slash);

  if (index_text.empty()) {
    throw std::runtime_error("OBJ face has an empty vertex index.");
  }

  const int obj_index = std::stoi(index_text);
  if (obj_index == 0) {
    throw std::runtime_error("OBJ vertex index 0 is invalid.");
  }

  const int index = obj_index > 0 ? obj_index - 1 : num_vertices + obj_index;
  if (index < 0 || index >= num_vertices) {
    throw std::runtime_error("OBJ face vertex index is out of range.");
  }

  return index;
}

double max_vertex_norm(const std::array<vector4f, NUM_DSF_VERTICES>& vertices) {
  double max_norm = 0.0;
  for (const vector4f& vertex : vertices) {
    max_norm = std::max(max_norm, static_cast<double>(norm(vertex)));
  }
  return max_norm;
}

void verify_scaling(
    const DSFVertexSet& vertex_set,
    ScalingMethod method,
    int p,
    float alpha,
    const std::array<vector4f, NUM_DSF_VERTICES>& scaled_vertices) {
  const double scaled_vmax = max_vertex_norm(scaled_vertices);
  const double float_max = std::numeric_limits<float>::max();
  bool valid = true;

  if (method == ScalingMethod::Raw) {
    valid = alpha == 1.0f;
  } else if (method == ScalingMethod::Max1) {
    valid = approximately_equal(scaled_vmax, 1.0, 1e-4);
  } else if (method == ScalingMethod::MaxFmax) {
    const double bound =
        static_cast<double>(NUM_DSF_VERTICES) * std::pow(scaled_vmax, p);
    valid =
        std::isfinite(bound) &&
        bound <= ETA * float_max * (1.0 + 1e-4);
  }

  std::cout << "[scaling][" << vertex_set.shape_type << "]["
            << scaling_method_name(method) << "][p=" << p
            << "] alpha=" << std::setprecision(8) << alpha
            << " vmax=" << scaled_vmax << '\n';

  if (!valid) {
    throw std::runtime_error("Scaling correctness check failed.");
  }
}

double mean_of(const std::vector<double>& values) {
  if (values.empty()) {
    return 0.0;
  }

  const double sum = std::accumulate(values.begin(), values.end(), 0.0);
  return sum / static_cast<double>(values.size());
}

double sample_std_of(const std::vector<double>& values, double mean) {
  if (values.size() < 2) {
    return 0.0;
  }

  double variance_sum = 0.0;
  for (double value : values) {
    const double delta = value - mean;
    variance_sum += delta * delta;
  }

  return std::sqrt(variance_sum / static_cast<double>(values.size() - 1));
}

double quantile_from_sorted(const std::vector<double>& sorted, double q) {
  if (sorted.empty()) {
    return 0.0;
  }

  const double position = q * static_cast<double>(sorted.size() - 1);
  const std::size_t lower = static_cast<std::size_t>(position);
  const std::size_t upper = std::min(lower + 1, sorted.size() - 1);
  const double weight = position - static_cast<double>(lower);
  return (1.0 - weight) * sorted[lower] + weight * sorted[upper];
}

Arguments parse_arguments(int argc, char** argv) {
  Arguments arguments;

  for (int i = 1; i < argc; ++i) {
    const std::string option = argv[i];
    if (i + 1 >= argc) {
      throw std::runtime_error("Missing value for command line option: " + option);
    }

    const std::string value = argv[++i];
    if (option == "--mesh") {
      arguments.mesh_path = value;
    } else if (option == "--samples") {
      arguments.num_samples = std::stoi(value);
    } else if (option == "--triangles") {
      arguments.triangles_per_sample = std::stoi(value);
    } else if (option == "--out_dir") {
      arguments.out_dir = value;
    } else {
      throw std::runtime_error("Unknown command line option: " + option);
    }
  }

  if (DEBUG_SMALL_RUN) {
    arguments.num_samples = 100;
    arguments.triangles_per_sample = 3;
  }

  if (arguments.num_samples <= 0 || arguments.triangles_per_sample <= 0) {
    throw std::runtime_error("--samples and --triangles must be positive.");
  }

  return arguments;
}

}  // namespace

std::vector<DSFVertexSet> load_dsf_vertices_csv(const std::string& path) {
  std::ifstream input(path);
  if (!input.is_open()) {
    throw std::runtime_error("Failed to open DSF vertex CSV: " + path);
  }

  std::unordered_map<std::string, std::size_t> shape_indices;
  std::vector<DSFVertexSet> vertex_sets;
  std::vector<std::array<bool, NUM_DSF_VERTICES>> seen_indices;
  std::vector<int> counts;
  std::vector<double> min_norms;
  std::vector<double> max_norms;

  for (const char* shape_name : SHAPE_ORDER) {
    DSFVertexSet vertex_set;
    vertex_set.shape_type = shape_name;
    shape_indices.emplace(shape_name, vertex_sets.size());
    vertex_sets.push_back(vertex_set);
    seen_indices.push_back({});
    counts.push_back(0);
    min_norms.push_back(std::numeric_limits<double>::infinity());
    max_norms.push_back(0.0);
  }

  std::string line;
  if (!std::getline(input, line)) {
    throw std::runtime_error("DSF vertex CSV is empty.");
  }

  int total_rows = 0;
  while (std::getline(input, line)) {
    if (line.empty()) {
      continue;
    }

    const DSFVertexRow row = parse_dsf_row(split_csv_line(line));
    const auto shape_it = shape_indices.find(row.shape_type);
    if (shape_it == shape_indices.end()) {
      throw std::runtime_error("Unexpected DSF shape_type: " + row.shape_type);
    }
    if (row.index < 1 || row.index > NUM_DSF_VERTICES) {
      throw std::runtime_error("DSF vertex index is outside 1..32.");
    }

    const std::size_t shape_index = shape_it->second;
    DSFVertexSet& vertex_set = vertex_sets[shape_index];
    const int vertex_index = row.index - 1;

    if (seen_indices[shape_index][vertex_index]) {
      throw std::runtime_error("Duplicate DSF vertex index in CSV.");
    }

    if (counts[shape_index] == 0) {
      vertex_set.R = row.R;
      vertex_set.log10_R = row.log10_R;
    } else if (
        !approximately_equal(vertex_set.R, row.R) ||
        !approximately_equal(vertex_set.log10_R, row.log10_R)) {
      throw std::runtime_error("R/log10_R is not constant within shape_type.");
    }

    const vector4f vertex(
        static_cast<float>(row.v_x),
        static_cast<float>(row.v_y),
        static_cast<float>(row.v_z));
    const double computed_norm = norm(vertex);
    if (!approximately_equal(computed_norm, row.norm_v, 1e-4)) {
      throw std::runtime_error("CSV norm_v does not match v coordinates.");
    }

    vertex_set.base_vertices[vertex_index] = vertex;
    seen_indices[shape_index][vertex_index] = true;
    ++counts[shape_index];
    ++total_rows;
    min_norms[shape_index] = std::min(min_norms[shape_index], computed_norm);
    max_norms[shape_index] = std::max(max_norms[shape_index], computed_norm);
  }

  if (total_rows != 160) {
    throw std::runtime_error("DSF vertex CSV must contain 160 data rows.");
  }

  for (std::size_t shape_index = 0; shape_index < vertex_sets.size(); ++shape_index) {
    if (counts[shape_index] != NUM_DSF_VERTICES) {
      throw std::runtime_error("Each DSF shape_type must contain 32 vertices.");
    }
    for (bool seen : seen_indices[shape_index]) {
      if (!seen) {
        throw std::runtime_error("DSF vertex indices 1..32 are incomplete.");
      }
    }

    const double computed_R = max_norms[shape_index] / min_norms[shape_index];
    if (!approximately_equal(computed_R, vertex_sets[shape_index].R, 1e-4)) {
      throw std::runtime_error("Computed DSF anisotropy R does not match CSV R.");
    }

    std::cout << "[DSF shape anisotropy][" << vertex_sets[shape_index].shape_type
              << "] R=" << vertex_sets[shape_index].R
              << " checked\n";
  }

  return vertex_sets;
}

MeshData load_obj_mesh(const std::string& path) {
  std::ifstream input(path);
  if (!input.is_open()) {
    throw std::runtime_error(
        "Failed to open OBJ mesh: " + path +
        ". Pass --mesh <path> if the default bunny OBJ is not present.");
  }

  MeshData mesh;
  std::string line;
  while (std::getline(input, line)) {
    std::istringstream stream(line);
    std::string type;
    stream >> type;

    if (type == "v") {
      float x = 0.0f;
      float y = 0.0f;
      float z = 0.0f;
      if (!(stream >> x >> y >> z)) {
        throw std::runtime_error("Failed to parse OBJ vertex.");
      }
      mesh.vertices.emplace_back(x, y, z);
    } else if (type == "f") {
      std::vector<int> face_indices;
      std::string token;
      while (stream >> token) {
        face_indices.push_back(
            parse_obj_face_index(token, static_cast<int>(mesh.vertices.size())));
      }
      if (face_indices.size() < 3) {
        throw std::runtime_error("OBJ face has fewer than 3 vertices.");
      }
      for (std::size_t i = 1; i + 1 < face_indices.size(); ++i) {
        mesh.triangles.push_back(
            {face_indices[0], face_indices[i], face_indices[i + 1]});
      }
    }
  }

  if (mesh.vertices.empty() || mesh.triangles.empty()) {
    throw std::runtime_error("OBJ mesh needs vertices and triangle faces.");
  }

  compute_triangle_centroids(mesh);
  return mesh;
}

void center_and_normalize_mesh(MeshData& mesh) {
  vector4f p_min(
      std::numeric_limits<float>::infinity(),
      std::numeric_limits<float>::infinity(),
      std::numeric_limits<float>::infinity());
  vector4f p_max(
      -std::numeric_limits<float>::infinity(),
      -std::numeric_limits<float>::infinity(),
      -std::numeric_limits<float>::infinity());

  for (const vector4f& vertex : mesh.vertices) {
    p_min.x = std::min(p_min.x, vertex.x);
    p_min.y = std::min(p_min.y, vertex.y);
    p_min.z = std::min(p_min.z, vertex.z);
    p_max.x = std::max(p_max.x, vertex.x);
    p_max.y = std::max(p_max.y, vertex.y);
    p_max.z = std::max(p_max.z, vertex.z);
  }

  const vector4f center = 0.5f * (p_min + p_max);
  float max_radius = 0.0f;
  for (vector4f& vertex : mesh.vertices) {
    vertex = vertex - center;
    max_radius = std::max(max_radius, norm(vertex));
  }
  if (!(max_radius > 0.0f) || !std::isfinite(max_radius)) {
    throw std::runtime_error("OBJ mesh has an invalid normalization radius.");
  }
  for (vector4f& vertex : mesh.vertices) {
    vertex = vertex / max_radius;
  }

  vector4f normalized_min(
      std::numeric_limits<float>::infinity(),
      std::numeric_limits<float>::infinity(),
      std::numeric_limits<float>::infinity());
  vector4f normalized_max(
      -std::numeric_limits<float>::infinity(),
      -std::numeric_limits<float>::infinity(),
      -std::numeric_limits<float>::infinity());
  float normalized_radius = 0.0f;
  for (const vector4f& vertex : mesh.vertices) {
    normalized_min.x = std::min(normalized_min.x, vertex.x);
    normalized_min.y = std::min(normalized_min.y, vertex.y);
    normalized_min.z = std::min(normalized_min.z, vertex.z);
    normalized_max.x = std::max(normalized_max.x, vertex.x);
    normalized_max.y = std::max(normalized_max.y, vertex.y);
    normalized_max.z = std::max(normalized_max.z, vertex.z);
    normalized_radius = std::max(normalized_radius, norm(vertex));
  }

  const vector4f normalized_center =
      0.5f * (normalized_min + normalized_max);
  if (norm(normalized_center) > 1e-4f ||
      !approximately_equal(normalized_radius, 1.0, 1e-4)) {
    throw std::runtime_error("Centered mesh correctness check failed.");
  }

  compute_triangle_centroids(mesh);
  std::cout << "[mesh] centered vertices=" << mesh.vertices.size()
            << " triangles=" << mesh.triangles.size()
            << " max_radius=" << normalized_radius << '\n';
}

std::vector<PoseSample> generate_pose_samples(
    int num_samples,
    float translation_range,
    unsigned int seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> translation_dist(
      -translation_range / 2.0f,
      translation_range / 2.0f);
  std::normal_distribution<float> gaussian(0.0f, 1.0f);

  std::vector<PoseSample> poses;
  poses.reserve(num_samples);
  for (int sample_index = 0; sample_index < num_samples; ++sample_index) {
    PoseSample pose;
    pose.translation = vector4f(
        translation_dist(rng),
        translation_dist(rng),
        translation_dist(rng));

    quaternion4f rotation;
    do {
      rotation = quaternion4f(
          gaussian(rng),
          gaussian(rng),
          gaussian(rng),
          gaussian(rng));
    } while (
        rotation.w * rotation.w + rotation.x * rotation.x +
            rotation.y * rotation.y + rotation.z * rotation.z <
        1e-20f);
    pose.rotation = normalize(rotation);
    poses.push_back(pose);
  }

  return poses;
}

std::vector<int> select_candidate_triangles(
    const MeshData& mesh,
    const PoseSample& pose,
    int triangles_per_sample,
    bool use_nearest_triangles,
    std::mt19937& rng) {
  const int triangle_count = static_cast<int>(mesh.triangles.size());
  const int selected_count = std::min(triangles_per_sample, triangle_count);

  if (use_nearest_triangles) {
    std::vector<std::pair<float, int>> distances;
    distances.reserve(mesh.triangle_centroids.size());
    for (int triangle_index = 0; triangle_index < triangle_count; ++triangle_index) {
      const vector4f delta =
          mesh.triangle_centroids[triangle_index] - pose.translation;
      distances.emplace_back(norm_squared(delta), triangle_index);
    }

    std::partial_sort(
        distances.begin(),
        distances.begin() + selected_count,
        distances.end());

    std::vector<int> selected;
    selected.reserve(selected_count);
    for (int i = 0; i < selected_count; ++i) {
      selected.push_back(distances[i].second);
    }
    return selected;
  }

  if (selected_count == 1) {
    std::uniform_int_distribution<int> triangle_dist(0, triangle_count - 1);
    return {triangle_dist(rng)};
  }

  std::vector<int> selected(triangle_count);
  std::iota(selected.begin(), selected.end(), 0);
  std::shuffle(selected.begin(), selected.end(), rng);
  selected.resize(selected_count);
  return selected;
}

float apply_scaling(
    ScalingMethod method,
    int p,
    const std::array<vector4f, 32>& base_vertices,
    std::array<vector4f, 32>& scaled_vertices) {
  const double vmax = max_vertex_norm(base_vertices);
  if (!(vmax > 0.0) || !std::isfinite(vmax)) {
    throw std::runtime_error("DSF vertices have an invalid maximum norm.");
  }

  double alpha = 1.0;
  if (method == ScalingMethod::Max1) {
    alpha = 1.0 / vmax;
  } else if (method == ScalingMethod::MaxFmax) {
    const double float_max = std::numeric_limits<float>::max();
    alpha =
        std::pow(ETA * float_max / static_cast<double>(NUM_DSF_VERTICES),
                 1.0 / static_cast<double>(p)) /
        vmax;
  }

  for (std::size_t i = 0; i < base_vertices.size(); ++i) {
    scaled_vertices[i] = static_cast<float>(alpha) * base_vertices[i];
  }

  return static_cast<float>(alpha);
}

double compute_underflow_rate(
    const std::array<vector4f, 32>& scaled_vertices,
    const vector4f& x_star,
    int p) {
  const double float_min = std::numeric_limits<float>::min();
  int active_count = 0;
  int underflow_count = 0;

  for (const vector4f& vertex : scaled_vertices) {
    const double a = std::max(static_cast<double>(dot(vertex, x_star)), 0.0);
    if (!(a > 0.0)) {
      continue;
    }

    ++active_count;
    const double term = std::pow(a, static_cast<double>(p));
    const float fp32_term = ::powf(static_cast<float>(a), static_cast<float>(p));
    if (term < float_min || fp32_term == 0.0f) {
      ++underflow_count;
    }
  }

  return active_count == 0
      ? 0.0
      : static_cast<double>(underflow_count) / static_cast<double>(active_count);
}

double compute_overflow_rate(
    const std::array<vector4f, 32>& scaled_vertices,
    const vector4f& x_star,
    int p) {
  const double float_max = std::numeric_limits<float>::max();

  for (const vector4f& vertex : scaled_vertices) {
    const double a = std::max(static_cast<double>(dot(vertex, x_star)), 0.0);
    const double term = std::pow(a, static_cast<double>(p));
    if (!std::isfinite(term) || term > float_max) {
      return 1.0;
    }
  }

  return 0.0;
}

SummaryMetrics compute_summary(const std::vector<SampleMetrics>& samples) {
  SummaryMetrics summary;
  if (samples.empty()) {
    return summary;
  }

  std::vector<double> optimalities;
  std::vector<double> underflow_rates;
  std::vector<double> overflow_rates;
  std::vector<double> success_rates;
  std::vector<double> residuals;
  std::vector<double> distances;
  optimalities.reserve(samples.size());
  underflow_rates.reserve(samples.size());
  overflow_rates.reserve(samples.size());
  success_rates.reserve(samples.size());
  residuals.reserve(samples.size());
  distances.reserve(samples.size());

  for (const SampleMetrics& sample : samples) {
    optimalities.push_back(finite_or_zero(sample.optimality));
    underflow_rates.push_back(finite_or_zero(sample.underflow_rate));
    overflow_rates.push_back(finite_or_zero(sample.overflow_rate));
    success_rates.push_back(finite_or_zero(sample.success_rate));
    residuals.push_back(finite_or_zero(sample.residual));
    distances.push_back(finite_or_zero(sample.distance));
  }

  summary.opt_mean = mean_of(optimalities);
  summary.opt_std = sample_std_of(optimalities, summary.opt_mean);
  std::sort(optimalities.begin(), optimalities.end());
  summary.opt_min = optimalities.front();
  summary.opt_Q1 = quantile_from_sorted(optimalities, 0.25);
  summary.opt_Q2 = quantile_from_sorted(optimalities, 0.50);
  summary.opt_Q3 = quantile_from_sorted(optimalities, 0.75);
  summary.opt_max = optimalities.back();

  summary.underflow_rate_mean = mean_of(underflow_rates);
  summary.underflow_rate_std =
      sample_std_of(underflow_rates, summary.underflow_rate_mean);
  summary.overflow_rate_mean = mean_of(overflow_rates);
  summary.overflow_rate_std =
      sample_std_of(overflow_rates, summary.overflow_rate_mean);
  summary.success_rate_mean = mean_of(success_rates);
  summary.residual_mean = mean_of(residuals);
  summary.distance_mean = mean_of(distances);
  return summary;
}

void write_summary_csv(
    std::ostream& output,
    const DSFVertexSet& vertex_set,
    ScalingMethod method,
    int p,
    int num_samples,
    int triangles_per_sample,
    float translation_range,
    const SummaryMetrics& summary) {
  output << std::setprecision(9) << vertex_set.shape_type << ','
         << vertex_set.R << ',' << vertex_set.log10_R << ','
         << scaling_method_name(method) << ',' << p << ','
         << num_samples << ',' << triangles_per_sample << ','
         << translation_range << ',' << ETA << ','
         << std::numeric_limits<float>::min() << ','
         << std::numeric_limits<float>::max() << ','
         << summary.opt_mean << ',' << summary.opt_std << ','
         << summary.opt_min << ',' << summary.opt_Q1 << ','
         << summary.opt_Q2 << ',' << summary.opt_Q3 << ','
         << summary.opt_max << ',' << summary.underflow_rate_mean << ','
         << summary.underflow_rate_std << ',' << summary.overflow_rate_mean
         << ',' << summary.overflow_rate_std << ','
         << summary.success_rate_mean << ',' << summary.residual_mean << ','
         << summary.distance_mean << '\n';
}

void write_samples_csv(
    std::ostream& output,
    const DSFVertexSet& vertex_set,
    ScalingMethod method,
    int p,
    const std::vector<PoseSample>& poses,
    const std::vector<SampleMetrics>& samples) {
  if (poses.size() != samples.size()) {
    throw std::runtime_error("Pose and sample metric counts do not match.");
  }

  for (std::size_t sample_index = 0; sample_index < samples.size(); ++sample_index) {
    const PoseSample& pose = poses[sample_index];
    const SampleMetrics& sample = samples[sample_index];
    output << std::setprecision(9) << vertex_set.shape_type << ','
           << vertex_set.R << ',' << vertex_set.log10_R << ','
           << scaling_method_name(method) << ',' << p << ','
           << sample_index << ',' << pose.translation.x << ','
           << pose.translation.y << ',' << pose.translation.z << ','
           << pose.rotation.w << ',' << pose.rotation.x << ','
           << pose.rotation.y << ',' << pose.rotation.z << ','
           << sample.optimality << ',' << sample.underflow_rate << ','
           << sample.overflow_rate << ',' << sample.success_rate << ','
           << sample.residual << ',' << sample.distance << '\n';
  }
}

}  // namespace evaluation_metrics
}  // namespace cuDSF

int main(int argc, char** argv) {
  using cuDSF::evaluation_metrics::MeshData;
  using cuDSF::evaluation_metrics::PoseSample;
  using cuDSF::evaluation_metrics::SampleMetrics;
  using cuDSF::evaluation_metrics::ScalingMethod;
  using Algorithm = cuDSF::experimental::trimesh_dsf32d::Algorithm;

  try {
    const auto arguments =
        cuDSF::evaluation_metrics::parse_arguments(argc, argv);
    const auto vertex_sets =
        cuDSF::evaluation_metrics::load_dsf_vertices_csv(
            cuDSF::evaluation_metrics::DEFAULT_DSF_CSV_PATH);
    MeshData mesh =
        cuDSF::evaluation_metrics::load_obj_mesh(arguments.mesh_path);
    cuDSF::evaluation_metrics::center_and_normalize_mesh(mesh);

    if (static_cast<int>(mesh.triangles.size()) <
        arguments.triangles_per_sample) {
      throw std::runtime_error(
          "Mesh has fewer triangles than --triangles requests.");
    }

    std::vector<PoseSample> poses =
        cuDSF::evaluation_metrics::generate_pose_samples(
            arguments.num_samples,
            cuDSF::evaluation_metrics::TRANSLATION_RANGE,
            cuDSF::evaluation_metrics::RNG_SEED);

    std::mt19937 triangle_rng(cuDSF::evaluation_metrics::RNG_SEED + 1u);
    for (PoseSample& pose : poses) {
      pose.candidate_triangles =
          cuDSF::evaluation_metrics::select_candidate_triangles(
              mesh,
              pose,
              arguments.triangles_per_sample,
              cuDSF::evaluation_metrics::USE_NEAREST_TRIANGLES,
              triangle_rng);
    }

    std::filesystem::create_directories(arguments.out_dir);
    const std::filesystem::path out_dir(arguments.out_dir);
    std::ofstream summary_output(out_dir / "optimality_summary.csv");
    std::ofstream sample_output(out_dir / "optimality_samples.csv");
    if (!summary_output.is_open() || !sample_output.is_open()) {
      throw std::runtime_error("Failed to create optimality CSV outputs.");
    }

    summary_output
        << "shape_type,R,log10_R,method,p,num_samples,triangles_per_sample,"
           "L,eta,Fmin,Fmax,opt_mean,opt_std,opt_min,opt_Q1,opt_Q2,opt_Q3,"
           "opt_max,underflow_rate_mean,underflow_rate_std,overflow_rate_mean,"
           "overflow_rate_std,success_rate_mean,residual_mean,distance_mean\n";
    sample_output
        << "shape_type,R,log10_R,method,p,sample_index,translation_x,"
           "translation_y,translation_z,quaternion_w,quaternion_x,"
           "quaternion_y,quaternion_z,opt_sample,underflow_rate_sample,"
           "overflow_rate_sample,success_rate_sample,residual_sample,"
           "distance_sample\n";

    Algorithm algorithm;
    for (const auto& vertex_set : vertex_sets) {
      for (ScalingMethod method : cuDSF::evaluation_metrics::SCALING_METHODS) {
        for (int p : cuDSF::evaluation_metrics::P_VALUES) {
          std::array<cuDSF::vector4f, 32> scaled_dsf_vertices;
          const float alpha =
              cuDSF::evaluation_metrics::apply_scaling(
                  method,
                  p,
                  vertex_set.base_vertices,
                  scaled_dsf_vertices);
          cuDSF::evaluation_metrics::verify_scaling(
              vertex_set, method, p, alpha, scaled_dsf_vertices);

          std::vector<SampleMetrics> sample_metrics;
          sample_metrics.reserve(poses.size());
          for (const PoseSample& pose : poses) {
            SampleMetrics sample_metric;
            const cuDSF::quaternion4f q_inv(
                pose.rotation.w,
                -pose.rotation.x,
                -pose.rotation.y,
                -pose.rotation.z);

            for (int triangle_index : pose.candidate_triangles) {
              const std::array<int, 3>& triangle =
                  mesh.triangles[triangle_index];
              const cuDSF::vector4f t1_local =
                  rotate(q_inv, mesh.vertices[triangle[0]] - pose.translation);
              const cuDSF::vector4f t2_local =
                  rotate(q_inv, mesh.vertices[triangle[1]] - pose.translation);
              const cuDSF::vector4f t3_local =
                  rotate(q_inv, mesh.vertices[triangle[2]] - pose.translation);

              const Algorithm::EvalResult result =
                  algorithm.solve_single_triangle_for_evaluation(
                      scaled_dsf_vertices,
                      32,
                      alpha * t1_local,
                      alpha * t2_local,
                      alpha * t3_local,
                      static_cast<double>(p));

              sample_metric.optimality +=
                  cuDSF::evaluation_metrics::finite_or_zero(result.optimality);
              sample_metric.underflow_rate +=
                  cuDSF::evaluation_metrics::compute_underflow_rate(
                      scaled_dsf_vertices, result.x_star, p);
              sample_metric.overflow_rate +=
                  cuDSF::evaluation_metrics::compute_overflow_rate(
                      scaled_dsf_vertices, result.x_star, p);
              sample_metric.success_rate += result.success ? 1.0 : 0.0;
              sample_metric.residual +=
                  cuDSF::evaluation_metrics::finite_or_zero(result.residual);
              sample_metric.distance +=
                  cuDSF::evaluation_metrics::finite_or_zero(result.distance);
            }

            const double divisor =
                static_cast<double>(pose.candidate_triangles.size());
            sample_metric.optimality /= divisor;
            sample_metric.underflow_rate /= divisor;
            sample_metric.overflow_rate /= divisor;
            sample_metric.success_rate /= divisor;
            sample_metric.residual /= divisor;
            sample_metric.distance /= divisor;
            sample_metrics.push_back(sample_metric);
          }

          const auto summary =
              cuDSF::evaluation_metrics::compute_summary(sample_metrics);
          cuDSF::evaluation_metrics::write_summary_csv(
              summary_output,
              vertex_set,
              method,
              p,
              arguments.num_samples,
              arguments.triangles_per_sample,
              cuDSF::evaluation_metrics::TRANSLATION_RANGE,
              summary);
          cuDSF::evaluation_metrics::write_samples_csv(
              sample_output, vertex_set, method, p, poses, sample_metrics);
          summary_output.flush();
          sample_output.flush();

          std::cout << '[' << vertex_set.shape_type << "]["
                    << cuDSF::evaluation_metrics::scaling_method_name(method)
                    << "][p=" << p << "] completed\n";
        }
      }
    }
  } catch (const std::exception& error) {
    std::cerr << "eval_optimality failed: " << error.what() << '\n';
    return 1;
  }

  return 0;
}