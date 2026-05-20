#pragma once

#include "CoreMinimal.h"
#include <cstdint>
struct FVec3
{
    float X = 0.0f;
    float Y = 0.0f;
    float Z = 0.0f;
};

struct FDSF_Parameter_Settings
{
    int32_t NUM_Initial_Directions = 1000;

    int32_t NUM_Tau_Iterations = 8;
    int32_t NUM_Newton_Iterations = 35;
    int32_t NUM_Alpha_Iterations = 30;

    float Tau_Decay_Rate = 0.2f;
    float Tau_Min = 1e-9f;

    float Jacobian_Eps = 1e-6f;
    float Residual_Eps = 1e-9f;
};

struct FDSF_Debug_Frame
{
    FVec3 X;
    FVec3 Lambda;
    FVec3 P_C;
    FVec3 P_T;
    FVec3 P_C_Minus_P_T;

    float Epigraph_T = 0.0f;
    float Residual = 0.0f;
    float Optimality = 0.0f;
    float Tau = 0.0f;

    int32 FrameIndex = 0;
    int32 OuterIteration = 0;
    int32 NewtonIteration = 0;

    bool bInitialCandidate = false;
};

struct FDSF_Result
{
    FVec3 X_Star;
    FVec3 Lambda;

    float Epigraph_T = 0.0f;

    FVec3 P_C;
    FVec3 P_T;
    FVec3 Slack;

    float Z[7] = { 0.0f };
    float F[7] = { 0.0f };

    float Residual = 0.0f;
    float Optimality = 0.0f;

    bool bSuccess = false;

    TArray<FDSF_Debug_Frame> DebugFrames;
};

class DSF_Mesh_Builder
{
public:
    FDSF_Result Solve(
        const FVec3* V,
        int32_t Num_V,
        const FVec3& T1,
        const FVec3& T2,
        const FVec3& T3,
        float p
    );

    void Set_Settings(const FDSF_Parameter_Settings& In_Settings);

    TArray<FVec3> Generate_DSF_Boundary_Points(
        const FVec3* V,
        int32_t Num_V,
        float p,
        int32 Num_Directions
    );

    static float Dot(const FVec3& A, const FVec3& B);
    static float Norm(const FVec3& A);
    static FVec3 Normalize(const FVec3& A);
    static FVec3 Cross(const FVec3& A, const FVec3& B);

    // DSF surface point: p_C(x) = gradient h_C(x)
    static FVec3 ComputeSupportPointConvexObject(
        const FVec3& X,
        const FVec3* V,
        int32_t Num_V,
        float p
    );

    static FVec3 ComputeSupportPointTriangleMesh(
        const FVec3& Lambda,
        const FVec3& T1,
        const FVec3& T2,
        const FVec3& T3
    );

    static float ComputeDSFValue(
        const FVec3& X,
        const FVec3* V,
        int32_t Num_V,
        float p
    );

private:
    FDSF_Parameter_Settings Settings;

private:
    static FVec3 RandomUnitVector();

    static void TangentBasisFromX(
        const FVec3& X,
        FVec3& B1,
        FVec3& B2
    );

    static float OptimalityMeasure(
        const FVec3& X,
        const FVec3& P_C,
        const FVec3& P_T
    );

private:
    void Add_Debug_Frame(
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
    );

    void Generate_F(
        float Out_F[7],
        const float Z[7],
        const FVec3* V,
        int32_t Num_V,
        const FVec3& T1,
        const FVec3& T2,
        const FVec3& T3,
        float p,
        float Tau
    );

    FVec3 Compute_Slack(
        const float Z[7],
        const FVec3* V,
        int32_t Num_V,
        const FVec3& T1,
        const FVec3& T2,
        const FVec3& T3,
        float p
    );

    FVec3 Epigraph_Values(
        const FVec3& X,
        const FVec3* V,
        int32_t Num_V,
        const FVec3& T1,
        const FVec3& T2,
        const FVec3& T3,
        float p
    );

    void Generate_Jacobian(
        float Out_J[7][7],
        const float Z[7],
        const FVec3* V,
        int32_t Num_V,
        const FVec3& T1,
        const FVec3& T2,
        const FVec3& T3,
        float p,
        float Tau
    );

    void Compute_Step(
        float Out_Delta_Z[7],
        const float J[7][7],
        const float F[7]
    );

    bool Check_Line_Search(
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
    );

    FVec3 Extract_X(const float Z[7]) const;
    FVec3 Extract_Lambda(const float Z[7]) const;
    float Extract_T(const float Z[7]) const;

    void Make_Z(
        float Out_Z[7],
        const FVec3& X,
        const FVec3& Lambda,
        float T
    ) const;

    float Vector_Norm(const float A[7]) const;
};
