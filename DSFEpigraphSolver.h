#pragma once

#include "DSFTypes.h"
#include "DSFMath.h"
#include "DSFSupportPoint.h"

class FDSF_Epigraph_Solver
{
    public:
        explicit FDSF_Epigraph_Solver(
            const FDSF_Parameter_Settings& In_Settings
        );

        FDSF_Result Solve(
            const FVec3* V,
            int32_t Num_V,
            const FVec3& T1,
            const FVec3& T2,
            const FVec3& T3,
            float p
        );

    private:
        FDSF_Parameter_Settings Settings;

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