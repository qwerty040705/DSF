#pragma once

#include "DSFTypes.h"

class FDSFMath
{
    public:
        static void TangentBasisFromX(
            const FVec3& X,
            FVec3& B1,
            FVec3& B2
        );

        static FVec3 RandomUnitVector();

        static float OptimalityMeasure(
            const FVec3& X,
            const FVec3& P_C,
            const FVec3& P_T
        );

        static float Dot(
            const FVec3& A,
            const FVec3& B
        );

        static float Norm(
            const FVec3& A
        );

        static float Sigma(
            const float* Values,
            int32_t Num_Values
        );
};