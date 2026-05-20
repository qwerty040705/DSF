#include <total.h>

#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <limits>

void DSF_Mesh_Builder::Set_Settings(const FDSF_Parameter_Settings& In_Settings)
{
    Settings = In_Settings;
}

float DSF_Mesh_Builder::Dot(const FVec3& A, const FVec3& B)
{
    return A.X * B.X + A.Y * B.Y + A.Z * B.Z;
}

float DSF_Mesh_Builder::Norm(const FVec3& A)
{
    return std::sqrt(Dot(A, A));
}

FVec3 DSF_Mesh_Builder::Normalize(const FVec3& A)
{
    const float N = Norm(A);

    if (N < 1e-12f || !std::isfinite(N))
    {
        return { 1.0f, 0.0f, 0.0f };
    }

    return { A.X / N, A.Y / N, A.Z / N };
}

FVec3 DSF_Mesh_Builder::Cross(const FVec3& A, const FVec3& B)
{
    return {
        A.Y * B.Z - A.Z * B.Y,
        A.Z * B.X - A.X * B.Z,
        A.X * B.Y - A.Y * B.X
    };
}

FVec3 DSF_Mesh_Builder::RandomUnitVector()
{
    float X = 0.0f;
    float Y = 0.0f;
    float Z = 0.0f;
    float N = 0.0f;

    do
    {
        X = 2.0f * static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX) - 1.0f;
        Y = 2.0f * static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX) - 1.0f;
        Z = 2.0f * static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX) - 1.0f;
        N = std::sqrt(X * X + Y * Y + Z * Z);
    } while (N < 1e-12f || !std::isfinite(N));

    return { X / N, Y / N, Z / N };
}

void DSF_Mesh_Builder::TangentBasisFromX(
    const FVec3& X,
    FVec3& B1,
    FVec3& B2
)
{
    FVec3 E = Normalize(X);

    FVec3 A;

    if (std::abs(Dot(E, { 1.0f, 0.0f, 0.0f })) < 0.9f)
    {
        A = { 1.0f, 0.0f, 0.0f };
    }
    else
    {
        A = { 0.0f, 1.0f, 0.0f };
    }

    const float AE = Dot(A, E);

    B1 = {
        A.X - AE * E.X,
        A.Y - AE * E.Y,
        A.Z - AE * E.Z
    };

    B1 = Normalize(B1);
    B2 = Normalize(Cross(E, B1));
}

float DSF_Mesh_Builder::OptimalityMeasure(
    const FVec3& X,
    const FVec3& P_C,
    const FVec3& P_T
)
{
    FVec3 D = {
        P_C.X - P_T.X,
        P_C.Y - P_T.Y,
        P_C.Z - P_T.Z
    };

    const float Denom = Norm(X) * Norm(D);

    if (Denom < 1e-12f || !std::isfinite(Denom))
    {
        return 0.0f;
    }

    FVec3 C = Cross(X, D);
    return Norm(C) / Denom;
}

float DSF_Mesh_Builder::ComputeDSFValue(
    const FVec3& X,
    const FVec3* V,
    int32_t Num_V,
    float p
)
{
    float Sum = 0.0f;

    for (int32_t i = 0; i < Num_V; ++i)
    {
        const float DotValue = Dot(V[i], X);
        const float PositiveValue = std::max(DotValue, 0.0f);

        if (PositiveValue <= 0.0f)
        {
            continue;
        }

        const float Term = std::pow(PositiveValue, p);

        if (std::isfinite(Term))
        {
            Sum += Term;
        }
        else
        {
            return std::numeric_limits<float>::infinity();
        }
    }

    if (Sum <= 0.0f || !std::isfinite(Sum))
    {
        return 0.0f;
    }

    return std::pow(Sum, 1.0f / p);
}

FVec3 DSF_Mesh_Builder::ComputeSupportPointConvexObject(
    const FVec3& X,
    const FVec3* V,
    int32_t Num_V,
    float p
)
{
    const float H = ComputeDSFValue(X, V, Num_V, p);

    if (H <= 1e-12f || !std::isfinite(H))
    {
        return { 0.0f, 0.0f, 0.0f };
    }

    FVec3 Sum = { 0.0f, 0.0f, 0.0f };

    for (int32_t i = 0; i < Num_V; ++i)
    {
        const float DotValue = Dot(V[i], X);
        const float PositiveValue = std::max(DotValue, 0.0f);

        if (PositiveValue <= 0.0f)
        {
            continue;
        }

        const float Weight = std::pow(PositiveValue, p - 1.0f);

        if (!std::isfinite(Weight))
        {
            continue;
        }

        Sum.X += Weight * V[i].X;
        Sum.Y += Weight * V[i].Y;
        Sum.Z += Weight * V[i].Z;
    }

    const float Denom = std::pow(H, p - 1.0f);

    if (Denom <= 1e-12f || !std::isfinite(Denom))
    {
        return { 0.0f, 0.0f, 0.0f };
    }

    return {
        Sum.X / Denom,
        Sum.Y / Denom,
        Sum.Z / Denom
    };
}

FVec3 DSF_Mesh_Builder::ComputeSupportPointTriangleMesh(
    const FVec3& Lambda,
    const FVec3& T1,
    const FVec3& T2,
    const FVec3& T3
)
{
    return {
        Lambda.X * T1.X + Lambda.Y * T2.X + Lambda.Z * T3.X,
        Lambda.X * T1.Y + Lambda.Y * T2.Y + Lambda.Z * T3.Y,
        Lambda.X * T1.Z + Lambda.Y * T2.Z + Lambda.Z * T3.Z
    };
}

TArray<FVec3> DSF_Mesh_Builder::Generate_DSF_Boundary_Points(
    const FVec3* V,
    int32_t Num_V,
    float p,
    int32 Num_Directions
)
{
    TArray<FVec3> Points;
    Points.Reserve(Num_Directions);

    for (int32 i = 0; i < Num_Directions; ++i)
    {
        FVec3 X = RandomUnitVector();
        FVec3 P = ComputeSupportPointConvexObject(X, V, Num_V, p);
        Points.Add(P);
    }

    return Points;
}

FDSF_Result DSF_Mesh_Builder::Solve(
    const FVec3* V,
    int32_t Num_V,
    const FVec3& T1,
    const FVec3& T2,
    const FVec3& T3,
    float p
)
{
    FDSF_Result Result;

    float Best_Value = std::numeric_limits<float>::infinity();
    FVec3 Best_X = { 1.0f, 0.0f, 0.0f };

    // 1) Initial candidate search.
    // This selected Best_X becomes DebugFrames[0].
    for (int32_t i = 0; i < Settings.NUM_Initial_Directions; ++i)
    {
        FVec3 X = RandomUnitVector();
        FVec3 Values = Epigraph_Values(X, V, Num_V, T1, T2, T3, p);

        const float Current_Value = std::max(Values.X, std::max(Values.Y, Values.Z));

        if (std::isfinite(Current_Value) && Current_Value < Best_Value)
        {
            Best_Value = Current_Value;
            Best_X = X;
        }
    }

    float Norm_X = Norm(Best_X);

    if (Norm_X < 1e-12f || !std::isfinite(Norm_X))
    {
        Best_X = { 1.0f, 0.0f, 0.0f };
        Norm_X = 1.0f;
    }

    FVec3 X = {
        Best_X.X / Norm_X,
        Best_X.Y / Norm_X,
        Best_X.Z / Norm_X
    };

    FVec3 Values = Epigraph_Values(X, V, Num_V, T1, T2, T3, p);
    float T = std::max(Values.X, std::max(Values.Y, Values.Z)) + 1e-2f;

    FVec3 Lambda = {
        1.0f / 3.0f,
        1.0f / 3.0f,
        1.0f / 3.0f
    };

    float Z[7];
    Make_Z(Z, X, Lambda, T);

    FVec3 Slack = Compute_Slack(Z, V, Num_V, T1, T2, T3, p);

    float Tau = std::max(
        (Lambda.X * Slack.X + Lambda.Y * Slack.Y + Lambda.Z * Slack.Z) / 3.0f,
        1e-4f
    );

    if (!std::isfinite(Tau) || Tau <= 0.0f)
    {
        Tau = 1e-4f;
    }

    bool bSuccess = false;

    Add_Debug_Frame(
        Result,
        Z,
        V,
        Num_V,
        T1,
        T2,
        T3,
        p,
        Tau,
        0,
        0,
        0.0f,
        true
    );

    // 2) Newton / barrier update.
    for (int32_t Outer = 0; Outer < Settings.NUM_Tau_Iterations; ++Outer)
    {
        Tau = std::max(Tau * Settings.Tau_Decay_Rate, Settings.Tau_Min);

        for (int32_t Iter = 0; Iter < Settings.NUM_Newton_Iterations; ++Iter)
        {
            float F[7];
            Generate_F(F, Z, V, Num_V, T1, T2, T3, p, Tau);

            bool bFinite_F = true;

            for (int32_t i = 0; i < 7; ++i)
            {
                if (!std::isfinite(F[i]))
                {
                    bFinite_F = false;
                    break;
                }
            }

            if (!bFinite_F)
            {
                break;
            }

            const float Residual_Norm = Vector_Norm(F);

            Add_Debug_Frame(
                Result,
                Z,
                V,
                Num_V,
                T1,
                T2,
                T3,
                p,
                Tau,
                Outer,
                Iter,
                Residual_Norm,
                false
            );

            if (Residual_Norm < Settings.Residual_Eps)
            {
                bSuccess = true;
                break;
            }

            float J[7][7];
            Generate_Jacobian(J, Z, V, Num_V, T1, T2, T3, p, Tau);

            float Delta_Z[7];
            Compute_Step(Delta_Z, J, F);

            bool bFinite_Delta = true;

            for (int32_t i = 0; i < 7; ++i)
            {
                if (!std::isfinite(Delta_Z[i]))
                {
                    bFinite_Delta = false;
                    break;
                }
            }

            if (!bFinite_Delta)
            {
                break;
            }

            const bool bAccepted = Check_Line_Search(
                Z,
                Delta_Z,
                F,
                V,
                Num_V,
                T1,
                T2,
                T3,
                p,
                Tau
            );

            if (!bAccepted)
            {
                // Very small fallback step.
                for (int32_t i = 0; i < 7; ++i)
                {
                    Z[i] += 1e-4f * Delta_Z[i];
                }
            }
        }
    }

    X = Extract_X(Z);
    Norm_X = Norm(X);

    if (Norm_X < 1e-12f || !std::isfinite(Norm_X))
    {
        X = { 1.0f, 0.0f, 0.0f };
    }
    else
    {
        X.X /= Norm_X;
        X.Y /= Norm_X;
        X.Z /= Norm_X;
    }

    Lambda = Extract_Lambda(Z);

    Lambda.X = std::max(Lambda.X, 0.0f);
    Lambda.Y = std::max(Lambda.Y, 0.0f);
    Lambda.Z = std::max(Lambda.Z, 0.0f);

    const float Lambda_Sum = Lambda.X + Lambda.Y + Lambda.Z;

    if (Lambda_Sum < 1e-14f || !std::isfinite(Lambda_Sum))
    {
        Lambda = {
            1.0f / 3.0f,
            1.0f / 3.0f,
            1.0f / 3.0f
        };
    }
    else
    {
        Lambda.X /= Lambda_Sum;
        Lambda.Y /= Lambda_Sum;
        Lambda.Z /= Lambda_Sum;
    }

    T = Extract_T(Z);

    Make_Z(Z, X, Lambda, T);

    FVec3 P_C = ComputeSupportPointConvexObject(X, V, Num_V, p);
    FVec3 P_T = ComputeSupportPointTriangleMesh(Lambda, T1, T2, T3);
    Slack = Compute_Slack(Z, V, Num_V, T1, T2, T3, p);

    float F_Final[7];
    Generate_F(F_Final, Z, V, Num_V, T1, T2, T3, p, Settings.Tau_Min);

    Result.X_Star = X;
    Result.Lambda = Lambda;
    Result.Epigraph_T = T;
    Result.P_C = P_C;
    Result.P_T = P_T;
    Result.Slack = Slack;
    Result.Residual = Vector_Norm(F_Final);
    Result.Optimality = OptimalityMeasure(X, P_C, P_T);
    Result.bSuccess = bSuccess || std::isfinite(Result.Residual);

    for (int32_t i = 0; i < 7; ++i)
    {
        Result.Z[i] = Z[i];
        Result.F[i] = F_Final[i];
    }

    Add_Debug_Frame(
        Result,
        Z,
        V,
        Num_V,
        T1,
        T2,
        T3,
        p,
        Settings.Tau_Min,
        Settings.NUM_Tau_Iterations,
        Settings.NUM_Newton_Iterations,
        Result.Residual,
        false
    );

    return Result;
}

void DSF_Mesh_Builder::Add_Debug_Frame(
    FDSF_Result& Result,
    const float Z[7],
    const FVec3* V,
    int32_t Num_V,
    const FVec3& T1,
    const FVec3& T2,
    const FVec3& T3,
    float p,
    float Tau,
    int32 OuterIteration,
    int32 NewtonIteration,
    float Residual,
    bool bInitialCandidate
)
{
    FVec3 X = Normalize(Extract_X(Z));
    FVec3 Lambda = Extract_Lambda(Z);

    Lambda.X = std::max(Lambda.X, 0.0f);
    Lambda.Y = std::max(Lambda.Y, 0.0f);
    Lambda.Z = std::max(Lambda.Z, 0.0f);

    const float LambdaSum = Lambda.X + Lambda.Y + Lambda.Z;

    if (LambdaSum > 1e-12f && std::isfinite(LambdaSum))
    {
        Lambda.X /= LambdaSum;
        Lambda.Y /= LambdaSum;
        Lambda.Z /= LambdaSum;
    }
    else
    {
        Lambda = { 1.0f / 3.0f, 1.0f / 3.0f, 1.0f / 3.0f };
    }

    FVec3 PC = ComputeSupportPointConvexObject(X, V, Num_V, p);
    FVec3 PT = ComputeSupportPointTriangleMesh(Lambda, T1, T2, T3);

    FDSF_Debug_Frame Frame;
    Frame.X = X;
    Frame.Lambda = Lambda;
    Frame.P_C = PC;
    Frame.P_T = PT;
    Frame.P_C_Minus_P_T = {
        PC.X - PT.X,
        PC.Y - PT.Y,
        PC.Z - PT.Z
    };
    Frame.Epigraph_T = Extract_T(Z);
    Frame.Residual = Residual;
    Frame.Optimality = OptimalityMeasure(X, PC, PT);
    Frame.Tau = Tau;
    Frame.FrameIndex = Result.DebugFrames.Num();
    Frame.OuterIteration = OuterIteration;
    Frame.NewtonIteration = NewtonIteration;
    Frame.bInitialCandidate = bInitialCandidate;

    Result.DebugFrames.Add(Frame);
}

void DSF_Mesh_Builder::Generate_F(
    float Out_F[7],
    const float Z[7],
    const FVec3* V,
    int32_t Num_V,
    const FVec3& T1,
    const FVec3& T2,
    const FVec3& T3,
    float p,
    float Tau
)
{
    FVec3 X = Extract_X(Z);
    FVec3 Lambda = Extract_Lambda(Z);

    if (Norm(X) < 1e-14f || !std::isfinite(Norm(X)))
    {
        X = { 1.0f, 0.0f, 0.0f };
    }

    FVec3 P_C = ComputeSupportPointConvexObject(X, V, Num_V, p);
    FVec3 P_T = ComputeSupportPointTriangleMesh(Lambda, T1, T2, T3);

    FVec3 B1;
    FVec3 B2;
    TangentBasisFromX(X, B1, B2);

    FVec3 Slack = Compute_Slack(Z, V, Num_V, T1, T2, T3, p);

    FVec3 Diff = {
        P_C.X - P_T.X,
        P_C.Y - P_T.Y,
        P_C.Z - P_T.Z
    };

    Out_F[0] = Dot(B1, Diff);
    Out_F[1] = Dot(B2, Diff);
    Out_F[2] = Dot(X, X) - 1.0f;
    Out_F[3] = Lambda.X + Lambda.Y + Lambda.Z - 1.0f;
    Out_F[4] = Lambda.X * Slack.X - Tau;
    Out_F[5] = Lambda.Y * Slack.Y - Tau;
    Out_F[6] = Lambda.Z * Slack.Z - Tau;
}

FVec3 DSF_Mesh_Builder::Compute_Slack(
    const float Z[7],
    const FVec3* V,
    int32_t Num_V,
    const FVec3& T1,
    const FVec3& T2,
    const FVec3& T3,
    float p
)
{
    FVec3 X = Extract_X(Z);
    const float T = Extract_T(Z);
    const float H = ComputeDSFValue(X, V, Num_V, p);

    return {
        T + Dot(T1, X) - H,
        T + Dot(T2, X) - H,
        T + Dot(T3, X) - H
    };
}

FVec3 DSF_Mesh_Builder::Epigraph_Values(
    const FVec3& X,
    const FVec3* V,
    int32_t Num_V,
    const FVec3& T1,
    const FVec3& T2,
    const FVec3& T3,
    float p
)
{
    const float H = ComputeDSFValue(X, V, Num_V, p);

    return {
        H - Dot(T1, X),
        H - Dot(T2, X),
        H - Dot(T3, X)
    };
}

void DSF_Mesh_Builder::Generate_Jacobian(
    float Out_J[7][7],
    const float Z[7],
    const FVec3* V,
    int32_t Num_V,
    const FVec3& T1,
    const FVec3& T2,
    const FVec3& T3,
    float p,
    float Tau
)
{
    for (int32_t Col = 0; Col < 7; ++Col)
    {
        float Z_Plus[7];
        float Z_Minus[7];

        for (int32_t i = 0; i < 7; ++i)
        {
            Z_Plus[i] = Z[i];
            Z_Minus[i] = Z[i];
        }

        const float Step = Settings.Jacobian_Eps * std::max(1.0f, std::abs(Z[Col]));

        Z_Plus[Col] += Step;
        Z_Minus[Col] -= Step;

        float F_Plus[7];
        float F_Minus[7];

        Generate_F(F_Plus, Z_Plus, V, Num_V, T1, T2, T3, p, Tau);
        Generate_F(F_Minus, Z_Minus, V, Num_V, T1, T2, T3, p, Tau);

        for (int32_t Row = 0; Row < 7; ++Row)
        {
            Out_J[Row][Col] = (F_Plus[Row] - F_Minus[Row]) / (2.0f * Step);

            if (!std::isfinite(Out_J[Row][Col]))
            {
                Out_J[Row][Col] = 0.0f;
            }
        }
    }
}

void DSF_Mesh_Builder::Compute_Step(
    float Out_Delta_Z[7],
    const float J[7][7],
    const float F[7]
)
{
    float A[7][8];

    for (int32_t i = 0; i < 7; ++i)
    {
        for (int32_t j = 0; j < 7; ++j)
        {
            A[i][j] = J[i][j];
        }

        A[i][7] = -F[i];
    }

    for (int32_t Col = 0; Col < 7; ++Col)
    {
        int32_t Pivot_Row = Col;
        float Pivot_Abs = std::abs(A[Col][Col]);

        for (int32_t Row = Col + 1; Row < 7; ++Row)
        {
            const float Candidate_Abs = std::abs(A[Row][Col]);

            if (Candidate_Abs > Pivot_Abs)
            {
                Pivot_Abs = Candidate_Abs;
                Pivot_Row = Row;
            }
        }

        if (Pivot_Abs < 1e-12f || !std::isfinite(Pivot_Abs))
        {
            for (int32_t i = 0; i < 7; ++i)
            {
                Out_Delta_Z[i] = 0.0f;
            }

            return;
        }

        if (Pivot_Row != Col)
        {
            for (int32_t j = Col; j < 8; ++j)
            {
                std::swap(A[Col][j], A[Pivot_Row][j]);
            }
        }

        const float Pivot = A[Col][Col];

        for (int32_t j = Col; j < 8; ++j)
        {
            A[Col][j] /= Pivot;
        }

        for (int32_t Row = 0; Row < 7; ++Row)
        {
            if (Row == Col)
            {
                continue;
            }

            const float Factor = A[Row][Col];

            for (int32_t j = Col; j < 8; ++j)
            {
                A[Row][j] -= Factor * A[Col][j];
            }
        }
    }

    for (int32_t i = 0; i < 7; ++i)
    {
        Out_Delta_Z[i] = A[i][7];
    }
}

bool DSF_Mesh_Builder::Check_Line_Search(
    float Z[7],
    const float Delta_Z[7],
    const float F_Old[7],
    const FVec3* V,
    int32_t Num_V,
    const FVec3& T1,
    const FVec3& T2,
    const FVec3& T3,
    float p,
    float Tau
)
{
    float Alpha = 1.0f;
    const float Old_Norm = Vector_Norm(F_Old);

    for (int32_t Iter = 0; Iter < Settings.NUM_Alpha_Iterations; ++Iter)
    {
        float Z_New[7];

        for (int32_t i = 0; i < 7; ++i)
        {
            Z_New[i] = Z[i] + Alpha * Delta_Z[i];
        }

        FVec3 X_New = Extract_X(Z_New);
        FVec3 Lambda_New = Extract_Lambda(Z_New);
        FVec3 Slack_New = Compute_Slack(Z_New, V, Num_V, T1, T2, T3, p);

        const bool bValid =
            Norm(X_New) > 1e-12f &&
            Lambda_New.X > 0.0f &&
            Lambda_New.Y > 0.0f &&
            Lambda_New.Z > 0.0f &&
            Slack_New.X > 0.0f &&
            Slack_New.Y > 0.0f &&
            Slack_New.Z > 0.0f;

        if (bValid)
        {
            float F_New[7];
            Generate_F(F_New, Z_New, V, Num_V, T1, T2, T3, p, Tau);

            const float New_Norm = Vector_Norm(F_New);

            if ((std::isfinite(New_Norm) && New_Norm < Old_Norm) || Alpha < 1e-5f)
            {
                for (int32_t i = 0; i < 7; ++i)
                {
                    Z[i] = Z_New[i];
                }

                return true;
            }
        }

        Alpha *= 0.5f;
    }

    return false;
}

FVec3 DSF_Mesh_Builder::Extract_X(const float Z[7]) const
{
    return { Z[0], Z[1], Z[2] };
}

FVec3 DSF_Mesh_Builder::Extract_Lambda(const float Z[7]) const
{
    return { Z[3], Z[4], Z[5] };
}

float DSF_Mesh_Builder::Extract_T(const float Z[7]) const
{
    return Z[6];
}

void DSF_Mesh_Builder::Make_Z(
    float Out_Z[7],
    const FVec3& X,
    const FVec3& Lambda,
    float T
) const
{
    Out_Z[0] = X.X;
    Out_Z[1] = X.Y;
    Out_Z[2] = X.Z;

    Out_Z[3] = Lambda.X;
    Out_Z[4] = Lambda.Y;
    Out_Z[5] = Lambda.Z;

    Out_Z[6] = T;
}

float DSF_Mesh_Builder::Vector_Norm(const float A[7]) const
{
    float Sum = 0.0f;

    for (int32_t i = 0; i < 7; ++i)
    {
        Sum += A[i] * A[i];
    }

    return std::sqrt(Sum);
}
