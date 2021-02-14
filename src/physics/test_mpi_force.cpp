#include "gtest/gtest.h"
#include <mpi.h>
#include <string>

#include "mpi_headers/mpi_test_env.h"
#include "mpi_headers/mpi_utils.h"

/* inclusion of source file is needed
 * since we want to test a static function  */
extern "C" {
#include "force.c"
}

TEST(pbc_test, inside) {
        ASSERT_DOUBLE_EQ(pbc(3.0, 10.0 * 0.5), 3.0);
        ASSERT_DOUBLE_EQ(pbc(-2.5, 10.0 * 0.5), -2.5);
}

TEST(pbc_test, outside) {
        ASSERT_DOUBLE_EQ(pbc(7.0, 10.0 * 0.5), -3.0);
        ASSERT_DOUBLE_EQ(pbc(-8.0, 10.0 * 0.5), 2.0);
}

TEST(MPIForceTestSingle, single) {
        mdsys_t *sys = new mdsys_t;
        sys->natoms = 1;
        sys->box = 10.0;
        sys->epsilon = 1;
        sys->sigma = 1;
        sys->rcut = 0;

        sys->r = new vec3_t[1]();
        sys->f = new vec3_t[1];

        arr_seg_t proc_seg;

        proc_seg.size = sys->natoms;
        proc_seg.idx = 0;
        sys->proc_seg = &proc_seg;

        ASSERT_TRUE(sys->proc_seg);

        force(sys);

        ASSERT_EQ(sys->f[0].x, 0.0);
        ASSERT_EQ(sys->f[0].y, 0.0);
        ASSERT_EQ(sys->f[0].z, 0.0);
        ASSERT_DOUBLE_EQ(sys->epot, 0.0);


        delete[] sys->f;
        delete[] sys->r;


        sys->proc_seg = nullptr;

        delete sys;
}

class MPI_ForceTest : public ::testing::TestWithParam<double> {

      protected:
        arr_seg_t proc_seg;
        int nprocs;
        int proc_id;
        mdsys_t *sys;
        int eps_param = GetParam();

        void SetUp() {
                nprocs = MPITestEnv::get_mpi_procs();
                proc_id = MPITestEnv::get_mpi_rank();

                ASSERT_EQ(nprocs, 3);

                sys = new mdsys_t;
                ASSERT_TRUE(sys);

                sys->proc_seg = &proc_seg;

                proc_seg.splitting = new int[nprocs]();
                ASSERT_TRUE(proc_seg.splitting);

                sys->natoms = 3;
                sys->epsilon = eps_param;
                sys->sigma = 1.0;
                sys->box = 10.0;
                sys->nprocs = nprocs;
                sys->proc_id = proc_id;

                sys->r = new vec3_t[3]();

                sys->f = new vec3_t[3];

                sys->r[0].x = -1.0;
                sys->r[1].x = 0.0;
                sys->r[2].x = 1.0;
        }

        void TearDown() {
                delete[] sys->r;
                delete[] sys->f;

                delete proc_seg.splitting;

                sys->proc_seg = nullptr;
                delete sys;
        }
};

TEST_P(MPI_ForceTest, shortrange) {

        ASSERT_DOUBLE_EQ(sys->natoms, 3);
        ASSERT_DOUBLE_EQ(sys->r[0].x, -1.0);
        ASSERT_DOUBLE_EQ(sys->r[1].x, 0.0);
        ASSERT_DOUBLE_EQ(sys->r[2].x, 1.0);
        ASSERT_DOUBLE_EQ(sys->r[0].y, 0.0);
        ASSERT_DOUBLE_EQ(sys->r[1].y, 0.0);
        ASSERT_DOUBLE_EQ(sys->r[2].y, 0.0);
        ASSERT_DOUBLE_EQ(sys->r[0].z, 0.0);
        ASSERT_DOUBLE_EQ(sys->r[1].z, 0.0);
        ASSERT_DOUBLE_EQ(sys->r[2].z, 0.0);

        init_segments(nprocs, proc_id, sys->proc_seg, sys->natoms);

        sys->rcut = 0.5;

        force(sys);

        double exp_epot_1 = 0.0;
        double exp_epot_2 = -eps_param * 63.0 / 1024.0;
        double exp_ff_1 = -eps_param * 24.0;
        double exp_ff_2 = eps_param * 93.0 / 512.0;

        for (int k = 0; k < sys->natoms; ++k) {
                EXPECT_DOUBLE_EQ(sys->f[k].x, 0.0);
                EXPECT_DOUBLE_EQ(sys->f[k].y, 0.0);
                EXPECT_DOUBLE_EQ(sys->f[k].z, 0.0);
        }
        EXPECT_DOUBLE_EQ(sys->epot, 0.0);
}

TEST_P(MPI_ForceTest, longrange) {

        ASSERT_DOUBLE_EQ(sys->natoms, 3);
        ASSERT_DOUBLE_EQ(sys->r[0].x, -1.0);
        ASSERT_DOUBLE_EQ(sys->r[1].x, 0.0);
        ASSERT_DOUBLE_EQ(sys->r[2].x, 1.0);
        ASSERT_DOUBLE_EQ(sys->r[0].y, 0.0);
        ASSERT_DOUBLE_EQ(sys->r[1].y, 0.0);
        ASSERT_DOUBLE_EQ(sys->r[2].y, 0.0);
        ASSERT_DOUBLE_EQ(sys->r[0].z, 0.0);
        ASSERT_DOUBLE_EQ(sys->r[1].z, 0.0);
        ASSERT_DOUBLE_EQ(sys->r[2].z, 0.0);

        init_segments(nprocs, proc_id, sys->proc_seg, sys->natoms);

        sys->rcut = 4.0;

        force(sys);

        double exp_epot_1 = 0.0;
        double exp_epot_2 = -eps_param * 63.0 / 1024.0;
        double exp_ff_1 = -eps_param * 24.0;
        double exp_ff_2 = eps_param * 93.0 / 512.0;

        for (int k = 0; k < sys->natoms; ++k) {
                EXPECT_DOUBLE_EQ(sys->f[k].y, 0.0);
                EXPECT_DOUBLE_EQ(sys->f[k].z, 0.0);
        }

        switch (sys->proc_id) {
        case 0:
                EXPECT_DOUBLE_EQ(sys->f[0].x, exp_ff_1 + exp_ff_2);
                EXPECT_DOUBLE_EQ(sys->f[1].x, -exp_ff_1);
                EXPECT_DOUBLE_EQ(sys->f[2].x, -exp_ff_2);
                EXPECT_DOUBLE_EQ(sys->epot, exp_epot_1 + exp_epot_2);
                break;
        case 1:
                EXPECT_DOUBLE_EQ(sys->f[0].x, 0.0);
                EXPECT_DOUBLE_EQ(sys->f[1].x, exp_ff_1);
                EXPECT_DOUBLE_EQ(sys->f[2].x, -exp_ff_1);
                EXPECT_DOUBLE_EQ(sys->epot, exp_epot_1);
                break;
        case 2:
                EXPECT_DOUBLE_EQ(sys->f[0].x, 0.0);
                EXPECT_DOUBLE_EQ(sys->f[1].x, 0.0);
                EXPECT_DOUBLE_EQ(sys->f[2].x, 0.0);
                EXPECT_DOUBLE_EQ(sys->epot, 0.0);
                break;
        default:
                break;
        }
}

INSTANTIATE_TEST_SUITE_P(MPI_ForceTest_parametric, MPI_ForceTest,
                         ::testing::Values(1.0));

int main(int argc, char **argv) {

        std::string command_line_arg(argc == 2 ? argv[1] : "");
        testing::InitGoogleTest(&argc, argv);
        ::testing::AddGlobalTestEnvironment(new MPITestEnv(command_line_arg));

        return RUN_ALL_TESTS();
}