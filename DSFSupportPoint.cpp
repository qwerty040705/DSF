#include <iostream>
#include <cmath>
#include <cstdlib>
#include "DSFTypes.h"
#include "DSFMath.h"
#include "DSFSupportPoint.h"

float FDSF_Value::Compute(
    const FVec3& X,
    const FVec3* V,
    int32_t Num_V,
    float p
)
{
    float Sum = 0.0f;

    for (int32_t i = 0; i < Num_V; ++i)
    {
        const float DotValue = FDSFMath::Dot(V[i], X);
        const float PositiveValue = std::max(DotValue, 0.0f);

        Sum += std::pow(PositiveValue, p);
    }

    if (Sum <= 0.0f)
    {
        return 0.0f;
    }

    return std::pow(Sum, 1.0f / p);
}

FVec3 Support_Point::Convex_Object(
    const FVec3& X,
    const FVec3* V,
    int32_t Num_V,
    float p
)
{
    const float H = FDSF_Value::Compute(X, V, Num_V, p);

    if (H <= 1e-12f)
    {
        return {0.0f, 0.0f, 0.0f};
    }

    FVec3 Sum = {0.0f, 0.0f, 0.0f};

    for (int32_t i = 0; i < Num_V; ++i)
    {
        const float DotValue = FDSFMath::Dot(V[i], X);
        const float PositiveValue = std::max(DotValue, 0.0f);
        const float Weight = std::pow(PositiveValue, p - 1.0f);

        Sum.X += Weight * V[i].X;
        Sum.Y += Weight * V[i].Y;
        Sum.Z += Weight * V[i].Z;
    }

    const float Denom = std::pow(H, p - 1.0f);

    if (Denom <= 1e-12f)
    {
        return {0.0f, 0.0f, 0.0f};
    }

    return {
        Sum.X / Denom,
        Sum.Y / Denom,
        Sum.Z / Denom
    };
}

FVec3 Support_Point::Triangle_Mesh(
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