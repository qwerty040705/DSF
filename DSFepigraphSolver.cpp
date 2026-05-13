#include "DSFEpigraphSolver.h"

#include <cmath>
#include <algorithm>

FDSF_Epigraph_Solver::FDSF_Epigraph_Solver(
    const FDSF_Parameter_Settings& In_Settings
)
    : Settings(In_Settings)
{
}

FDSF_Result FDSF_Epigraph_Solver::Solve(
    const FVec3* V,
    int32_t Num_V,
    const FVec3& T1,
    const FVec3& T2,
    const FVec3& T3,
    float p
)
{
    FDSF_Result Result;

    float Best_Value = INFINITY;
    FVec3 Best_X = {1.0f, 0.0f, 0.0f};

    for (int32_t i = 0; i < Settings.NUM_Initial_Directions; ++i)
    {
        FVec3 X = FDSFMath::RandomUnitVector();
        FVec3 Values = Epigraph_Values(X, V, Num_V, T1, T2, T3, p);

        float Current_Value = std::max(Values.X, std::max(Values.Y, Values.Z));

        if (std::isfinite(Current_Value) && Current_Value < Best_Value)
        {
            Best_Value = Current_Value;
            Best_X = X;
        }
    }

    float Norm_X = FDSFMath::Norm(Best_X);

    if (Norm_X < 1e-12f)
    {
        Best_X = {1.0f, 0.0f, 0.0f};
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

    bool bSuccess = false;

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

            float Residual_Norm = Vector_Norm(F);

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

            bool bAccepted = Check_Line_Search(
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
                for (int32_t i = 0; i < 7; ++i)
                {
                    Z[i] += 1e-4f * Delta_Z[i];
                }
            }
        }
    }

    X = Extract_X(Z);
    Norm_X = FDSFMath::Norm(X);

    if (Norm_X < 1e-12f || !std::isfinite(Norm_X))
    {
        X = {1.0f, 0.0f, 0.0f};
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

    float Lambda_Sum = Lambda.X + Lambda.Y + Lambda.Z;

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

    FVec3 P_C = Support_Point::Convex_Object(X, V, Num_V, p);
    FVec3 P_T = Support_Point::Triangle_Mesh(Lambda, T1, T2, T3);
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
    Result.Optimality = FDSFMath::OptimalityMeasure(X, P_C, P_T);
    Result.bSuccess = bSuccess || std::isfinite(Result.Residual);

    for (int32_t i = 0; i < 7; ++i)
    {
        Result.Z[i] = Z[i];
        Result.F[i] = F_Final[i];
    }

    return Result;
}

void FDSF_Epigraph_Solver::Generate_F(
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

    float Norm_X = FDSFMath::Norm(X);

    if (Norm_X < 1e-14f || !std::isfinite(Norm_X))
    {
        X = {1.0f, 0.0f, 0.0f};
    }

    FVec3 P_C = Support_Point::Convex_Object(X, V, Num_V, p);
    FVec3 P_T = Support_Point::Triangle_Mesh(Lambda, T1, T2, T3);

    FVec3 B1;
    FVec3 B2;
    FDSFMath::TangentBasisFromX(X, B1, B2);

    FVec3 Slack = Compute_Slack(Z, V, Num_V, T1, T2, T3, p);

    FVec3 Diff = {
        P_C.X - P_T.X,
        P_C.Y - P_T.Y,
        P_C.Z - P_T.Z
    };

    Out_F[0] = FDSFMath::Dot(B1, Diff);
    Out_F[1] = FDSFMath::Dot(B2, Diff);
    Out_F[2] = FDSFMath::Dot(X, X) - 1.0f;
    Out_F[3] = Lambda.X + Lambda.Y + Lambda.Z - 1.0f;
    Out_F[4] = Lambda.X * Slack.X - Tau;
    Out_F[5] = Lambda.Y * Slack.Y - Tau;
    Out_F[6] = Lambda.Z * Slack.Z - Tau;
}

FVec3 FDSF_Epigraph_Solver::Compute_Slack(
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
    float T = Extract_T(Z);
    float H = FDSF_Value::Compute(X, V, Num_V, p);

    return {
        T + FDSFMath::Dot(T1, X) - H,
        T + FDSFMath::Dot(T2, X) - H,
        T + FDSFMath::Dot(T3, X) - H
    };
}

FVec3 FDSF_Epigraph_Solver::Epigraph_Values(
    const FVec3& X,
    const FVec3* V,
    int32_t Num_V,
    const FVec3& T1,
    const FVec3& T2,
    const FVec3& T3,
    float p
)
{
    float H = FDSF_Value::Compute(X, V, Num_V, p);

    return {
        H - FDSFMath::Dot(T1, X),
        H - FDSFMath::Dot(T2, X),
        H - FDSFMath::Dot(T3, X)
    };
}

void FDSF_Epigraph_Solver::Generate_Jacobian(
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

        float Step = Settings.Jacobian_Eps * std::max(1.0f, std::abs(Z[Col]));

        Z_Plus[Col] += Step;
        Z_Minus[Col] -= Step;

        float F_Plus[7];
        float F_Minus[7];

        Generate_F(F_Plus, Z_Plus, V, Num_V, T1, T2, T3, p, Tau);
        Generate_F(F_Minus, Z_Minus, V, Num_V, T1, T2, T3, p, Tau);

        for (int32_t Row = 0; Row < 7; ++Row)
        {
            Out_J[Row][Col] = (F_Plus[Row] - F_Minus[Row]) / (2.0f * Step);
        }
    }
}

void FDSF_Epigraph_Solver::Compute_Step(
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
            float Candidate_Abs = std::abs(A[Row][Col]);

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

        float Pivot = A[Col][Col];

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

            float Factor = A[Row][Col];

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

bool FDSF_Epigraph_Solver::Check_Line_Search(
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
    float Old_Norm = Vector_Norm(F_Old);

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

        bool bValid =
            FDSFMath::Norm(X_New) > 1e-12f &&
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

            float New_Norm = Vector_Norm(F_New);

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

FVec3 FDSF_Epigraph_Solver::Extract_X(const float Z[7]) const
{
    return {Z[0], Z[1], Z[2]};
}

FVec3 FDSF_Epigraph_Solver::Extract_Lambda(const float Z[7]) const
{
    return {Z[3], Z[4], Z[5]};
}

float FDSF_Epigraph_Solver::Extract_T(const float Z[7]) const
{
    return Z[6];
}

void FDSF_Epigraph_Solver::Make_Z(
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

float FDSF_Epigraph_Solver::Vector_Norm(const float A[7]) const
{
    float Sum = 0.0f;

    for (int32_t i = 0; i < 7; ++i)
    {
        Sum += A[i] * A[i];
    }

    return std::sqrt(Sum);
}