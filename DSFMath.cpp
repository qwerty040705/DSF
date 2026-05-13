#include "DSFMath.h"

#include <cmath>
#include <cstdlib>

float FDSFMath::Dot(const FVec3& A, const FVec3& B)
{
    return A.X * B.X + A.Y * B.Y + A.Z * B.Z;
}

float FDSFMath::Norm(const FVec3& A)
{
    return std::sqrt(Dot(A, A));
}

float FDSFMath::Sigma(const float* Values, int32_t Num_Values)
{
    float Sum = 0.0f;

    for (int32_t i = 0; i < Num_Values; ++i)
    {
        Sum += Values[i];
    }

    return Sum;
}

void FDSFMath::TangentBasisFromX(const FVec3& X, FVec3& B1, FVec3& B2)
{
    FVec3 E = X;
    float NormX = Norm(E);

    if (NormX < 1e-14f)
    {
        E = {1.0f, 0.0f, 0.0f};
        NormX = 1.0f;
    }

    E.X /= NormX;
    E.Y /= NormX;
    E.Z /= NormX;

    FVec3 A;

    if (std::abs(Dot(E, {1.0f, 0.0f, 0.0f})) < 0.9f)
    {
        A = {1.0f, 0.0f, 0.0f};
    }
    else
    {
        A = {0.0f, 1.0f, 0.0f};
    }

    const float AE = Dot(A, E);

    B1 = {
        A.X - AE * E.X,
        A.Y - AE * E.Y,
        A.Z - AE * E.Z
    };

    const float NormB1 = Norm(B1);

    if (NormB1 > 1e-14f)
    {
        B1.X /= NormB1;
        B1.Y /= NormB1;
        B1.Z /= NormB1;
    }

    B2 = {
        E.Y * B1.Z - E.Z * B1.Y,
        E.Z * B1.X - E.X * B1.Z,
        E.X * B1.Y - E.Y * B1.X
    };

    const float NormB2 = Norm(B2);

    if (NormB2 > 1e-14f)
    {
        B2.X /= NormB2;
        B2.Y /= NormB2;
        B2.Z /= NormB2;
    }
}

FVec3 FDSFMath::RandomUnitVector()
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
    }
    while (N < 1e-12f);

    return {X / N, Y / N, Z / N};
}

float FDSFMath::OptimalityMeasure(const FVec3& X, const FVec3& P_C, const FVec3& P_T)
{
    FVec3 D = {
        P_C.X - P_T.X,
        P_C.Y - P_T.Y,
        P_C.Z - P_T.Z
    };

    FVec3 Cross = {
        X.Y * D.Z - X.Z * D.Y,
        X.Z * D.X - X.X * D.Z,
        X.X * D.Y - X.Y * D.X
    };

    const float Denom = Norm(X) * Norm(D);

    if (Denom < 1e-12f)
    {
        return 0.0f;
    }

    return Norm(Cross) / Denom;
}