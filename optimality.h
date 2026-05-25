#pragma once

#include <array>
#include <iosfwd>
#include <random>
#include <string>
#include <vector>

#include "global/include/arithmetic/quaternion4f.cuh"
#include "global/include/arithmetic/vector4f.cuh"

namespace cuDSF {
namespace evaluation_metrics {

enum class ScalingMethod { Raw, Max1, MaxFmax };

struct DSFVertexRow {
  std::string shape_type;
  double R = 0.0;
  double log10_R = 0.0;
  int index = 0;
  double r_i = 0.0;
  double u_x = 0.0;
  double u_y = 0.0;
  double u_z = 0.0;
  double v_x = 0.0;
  double v_y = 0.0;
  double v_z = 0.0;
  double norm_v = 0.0;
};

struct DSFVertexSet {
  std::string shape_type;
  double R = 0.0;
  double log10_R = 0.0;
  std::array<vector4f, 32> base_vertices;
};

struct MeshData {
  std::vector<vector4f> vertices;
  std::vector<std::array<int, 3>> triangles;
  std::vector<vector4f> triangle_centroids;
};

struct PoseSample {
  vector4f translation;
  quaternion4f rotation;
  std::vector<int> candidate_triangles;
};

struct SampleMetrics {
  double optimality = 0.0;
  double underflow_rate = 0.0;
  double overflow_rate = 0.0;
  double success_rate = 0.0;
  double residual = 0.0;
  double distance = 0.0;
};

struct SummaryMetrics {
  double opt_mean = 0.0;
  double opt_std = 0.0;
  double opt_min = 0.0;
  double opt_Q1 = 0.0;
  double opt_Q2 = 0.0;
  double opt_Q3 = 0.0;
  double opt_max = 0.0;
  double underflow_rate_mean = 0.0;
  double underflow_rate_std = 0.0;
  double overflow_rate_mean = 0.0;
  double overflow_rate_std = 0.0;
  double success_rate_mean = 0.0;
  double residual_mean = 0.0;
  double distance_mean = 0.0;
};

std::vector<DSFVertexSet> load_dsf_vertices_csv(const std::string& path);
MeshData load_obj_mesh(const std::string& path);
void center_and_normalize_mesh(MeshData& mesh);
std::vector<PoseSample> generate_pose_samples(
    int num_samples,
    float translation_range,
    unsigned int seed);
std::vector<int> select_candidate_triangles(
    const MeshData& mesh,
    const PoseSample& pose,
    int triangles_per_sample,
    bool use_nearest_triangles,
    std::mt19937& rng);
float apply_scaling(
    ScalingMethod method,
    int p,
    const std::array<vector4f, 32>& base_vertices,
    std::array<vector4f, 32>& scaled_vertices);
double compute_underflow_rate(
    const std::array<vector4f, 32>& scaled_vertices,
    const vector4f& x_star,
    int p);
double compute_overflow_rate(
    const std::array<vector4f, 32>& scaled_vertices,
    const vector4f& x_star,
    int p);
SummaryMetrics compute_summary(const std::vector<SampleMetrics>& samples);
void write_summary_csv(
    std::ostream& output,
    const DSFVertexSet& vertex_set,
    ScalingMethod method,
    int p,
    int num_samples,
    int triangles_per_sample,
    float translation_range,
    const SummaryMetrics& summary);
void write_samples_csv(
    std::ostream& output,
    const DSFVertexSet& vertex_set,
    ScalingMethod method,
    int p,
    const std::vector<PoseSample>& poses,
    const std::vector<SampleMetrics>& samples);

}  // namespace evaluation_metrics
}  // namespace cuDSF