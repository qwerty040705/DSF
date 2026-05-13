#pragma once

#include "DSFTypes.h"

class FDSF_Value
{
    public:
        static float Compute(
            const FVec3& X,
            const FVec3* V,
            int32_t Num_V,
            float p
        );
};

class Support_Point
{
    public:
        static FVec3 Convex_Object(
            const FVec3& X,
            const FVec3* V,
            int32_t Num_V,
            float p
        );

        static FVec3 Triangle_Mesh(
            const FVec3& Lambda,
            const FVec3& T1,
            const FVec3& T2,
            const FVec3& T3
        );
};