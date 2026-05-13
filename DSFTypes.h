#pragma once

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

struct FDSF_Result
{
    FVec3 X_Star;
    FVec3 Lambda;

    float Epigraph_T = 0.0f;

    FVec3 P_C;
    FVec3 P_T;
    FVec3 Slack;

    float Z[7] = {0.0f};
    float F[7] = {0.0f};

    float Residual = 0.0f;
    float Optimality = 0.0f;

    bool bSuccess = false;
};