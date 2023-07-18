/**
 \file 		abyfloat.cpp
 \author	daniel.demmler@ec-spride.de
 \copyright	ABY - A Framework for Efficient Mixed-protocol Secure Two-party Computation
 Copyright (C) 2019 Engineering Cryptographic Protocols Group, TU Darmstadt
			This program is free software: you can redistribute it and/or modify
			it under the terms of the GNU Lesser General Public License as published
			by the Free Software Foundation, either version 3 of the License, or
			(at your option) any later version.
			ABY is distributed in the hope that it will be useful,
			but WITHOUT ANY WARRANTY; without even the implied warranty of
			MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
			GNU Lesser General Public License for more details.
			You should have received a copy of the GNU Lesser General Public License
			along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

#include <ENCRYPTO_utils/crypto/crypto.h>
#include <ENCRYPTO_utils/parse_options.h>
#include "../../abycore/aby/abyparty.h"
#include "../../abycore/circuit/share.h"
#include "../../abycore/circuit/booleancircuits.h"
#include "../../abycore/circuit/arithmeticcircuits.h"
#include "../../abycore/sharing/sharing.h"
#include <cassert>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <random>
#include <fstream>




void read_test_options(int32_t *argcp, char ***argvp, e_role *role,
					   uint32_t *bitlen, uint32_t *nvals, uint32_t *secparam, std::string *address,
					   uint16_t *port, int32_t *test_op, uint32_t *test_bit, double *fpa, double *fpb, std::string *inputfile, std::string *outputfile)
{

	uint32_t int_role = 0, int_port = 0, int_testbit = 0;

	parsing_ctx options[] =
		{{(void *)&int_role, T_NUM, "r", "Role: 0/1", true, false},
		 {(void *)&int_testbit, T_NUM, "i", "test bit", false, false},
		 {(void *)nvals, T_NUM, "n", "Number of parallel operation elements", false, false},
		 {(void *)bitlen, T_NUM, "b", "Bit-length, default 32", false, false},
		 {(void *)secparam, T_NUM, "s", "Symmetric Security Bits, default: 128", false, false},
		 {(void *)address, T_STR, "a", "IP-address, default: localhost", false, false},
		 {(void *)&int_port, T_NUM, "p", "Port, default: 7766", false, false},
		 {(void *)test_op, T_NUM, "t", "Single test (leave out for all operations), default: off", false, false},
		 {(void *)fpa, T_DOUBLE, "x", "FP a", false, false},
		 {(void *)fpb, T_DOUBLE, "y", "FP b", false, false},
		 {(void *)inputfile, T_STR, "f", "Input file containing face embeddings", true, false},
		 {(void *)outputfile, T_STR, "o", "Output file containing cos sim result", false, false}
		};

	if (!parse_options(argcp, argvp, options,
					   sizeof(options) / sizeof(parsing_ctx)))
	{
		print_usage(*argvp[0], options, sizeof(options) / sizeof(parsing_ctx));
		std::cout << "Exiting" << std::endl;
		exit(0);
	}

	assert(int_role < 2);
	*role = (e_role)int_role;

	if (int_port != 0)
	{
		assert(int_port < 1 << (sizeof(uint16_t) * 8));
		*port = (uint16_t)int_port;
	}

	*test_bit = int_testbit;
}

void test_verilog_add64_SIMD(e_role role, const std::string &address, uint16_t port, seclvl seclvl, uint32_t nvals, uint32_t nthreads,
							 e_mt_gen_alg mt_alg, e_sharing sharing, double afp, double bfp, std::string inputfile, std::string outputfile)
{

	// for addition we operate on doubles, so set bitlen to 64 bits
	uint32_t bitlen = 64;

	// two arrays of real-world embeddings
	std::vector<double> xembeddings;
	std::vector<double> yembeddings;

	// array for the Sy<role> share
	std::vector<double> share_embeddings;

	std::cout << "INPUT FILE NAME: " << inputfile << std::endl;
	// reading the non-xored embeddings, i.e. current face and database face

	std::fstream infile(inputfile);

	double x, y;

	while (infile >> x >> y) {
		//std::cout << "x: " << x << " | y: "<< y << std::endl;
		xembeddings.push_back(x);
		yembeddings.push_back(y);
	}

	assert(xembeddings.size() == nvals);
	assert(yembeddings.size() == nvals);

	// reading the xored embedding, i.e. either Sy<0> or Sy<1> depending on the role

	char *fname = (char *) malloc(150); // file name buffer 
    sprintf(fname, "share%d.txt", role);

	std::fstream infile_share(fname);

	double z;

	while(infile_share >> z) {
		share_embeddings.push_back(z);
	}

	assert(share_embeddings.size() == nvals);


	std::string circuit_dir = "../../bin/circ/";

	ABYParty *party = new ABYParty(role, address, port, seclvl, bitlen, nthreads, mt_alg, 100000, circuit_dir);

	std::vector<Sharing *> &sharings = party->GetSharings();

	BooleanCircuit *bc = (BooleanCircuit *)sharings[S_BOOL]->GetCircuitBuildRoutine();

	// arrays of integer pointers to doubles
	uint64_t xvals[nvals];
	uint64_t yvals[nvals];
	uint64_t sharevals[nvals];


	// verification in plaintext
	double ver_x_times_y[nvals];
	double ver_x_times_x[nvals];
	double ver_y_times_y[nvals];
	double ver_x_dot_y = 0;
	double ver_norm_x = 0;
	double ver_norm_y = 0;

	// S_c(X,Y) = (X \dot Y) / (norm(X) * norm(Y))


	// init for random values
	double lower_bound = 0;
	double upper_bound = 1;
	std::uniform_real_distribution<double> unif(lower_bound, upper_bound);
	//std::random_device r;
	std::default_random_engine re;
	share **shr_server_set, **shr_client_set, **shr_out;

	for (uint32_t i = 0; i < nvals; i++)
	{
		double current_x = xembeddings[i];
		double current_y = yembeddings[i];
		double current_share = share_embeddings[i];

		uint64_t *xptr = (uint64_t *)&current_x;
		uint64_t *yptr = (uint64_t *)&current_y;
		uint64_t *shareptr = (uint64_t *)&current_share;

		xvals[i] = *xptr;
		yvals[i] = *yptr;
		sharevals[i] = *shareptr;

		ver_x_times_y[i] = current_x * current_y;
		ver_x_dot_y += ver_x_times_y[i];

		ver_x_times_x[i] = current_x * current_x;
		ver_y_times_y[i] = current_y * current_y;
		ver_norm_x += ver_x_times_x[i];
		ver_norm_y += ver_y_times_y[i];
 

		// shr_server_set[i] = yc->PutSIMDINGate(bitlen, xvals[i], 1, SERVER);
		// shr_client_set[i] = yc->PutSIMDINGate(bitlen, yvals[i], 1, CLIENT);
	}

	ver_norm_x = sqrt(ver_norm_x);
	ver_norm_y = sqrt(ver_norm_y);

	double ver_cos_dist = 1 - (ver_x_dot_y / (ver_norm_x * ver_norm_y));


	// INPUTS

	share *s_xin[nvals];
	share *s_yin[nvals];
	for (int i = 0; i < nvals; i++) {
		if (role == SERVER) {
			s_xin[i] = bc->PutINGate(xvals[i], bitlen, SERVER);
		} else { // role == CLIENT
			s_xin[i] = bc->PutDummyINGate(bitlen);
		}
		s_yin[i] = bc->PutSharedINGate(sharevals[i], bitlen);
	}


	// computing x \dot y
	share *s_x_times_y[nvals];
	s_x_times_y[0] = bc->PutFPGate(s_xin[0], s_yin[0], MUL, bitlen, nvals, no_status);
	share *s_x_dot_y = s_x_times_y[0];

	for (int i = 1; i < nvals; i++) {
		s_x_times_y[i] = bc->PutFPGate(s_xin[i], s_yin[i], MUL, bitlen, nvals, no_status);
		s_x_dot_y = bc->PutFPGate(s_x_dot_y, s_x_times_y[i], ADD, bitlen, nvals, no_status);
	}
	share *s_x_dot_y_out = bc->PutOUTGate(s_x_dot_y, SERVER);

	// computing norm(x)
	share *s_x_times_x[nvals];
	s_x_times_x[0] = bc->PutFPGate(s_xin[0], s_xin[0], MUL, bitlen, nvals, no_status);
	share *s_x_dot_x = s_x_times_x[0];

	for (int i = 1; i < nvals; i++) {
		s_x_times_x[i] = bc->PutFPGate(s_xin[i], s_xin[i], MUL, bitlen, nvals, no_status);
		s_x_dot_x = bc->PutFPGate(s_x_dot_x, s_x_times_x[i], ADD, bitlen, nvals, no_status);
	}
	share *s_norm_x = bc->PutFPGate(s_x_dot_x, SQRT);
	share *s_norm_x_out = bc->PutOUTGate(s_norm_x, SERVER);


	// computing norm(y)
	share *s_y_times_y[nvals];
	s_y_times_y[0] = bc->PutFPGate(s_yin[0], s_yin[0], MUL, bitlen, nvals, no_status);
	share *s_y_dot_y = s_y_times_y[0];

	for (int i = 1; i < nvals; i++) {
		s_y_times_y[i] = bc->PutFPGate(s_yin[i], s_yin[i], MUL, bitlen, nvals, no_status);
		s_y_dot_y = bc->PutFPGate(s_y_dot_y, s_y_times_y[i], ADD, bitlen, nvals, no_status);
	}
	share *s_norm_y = bc->PutFPGate(s_y_dot_y, SQRT);
	share *s_norm_y_out = bc->PutOUTGate(s_norm_y, SERVER);


	// computing norm(y)

	share *s_norm_x_times_norm_y = bc->PutFPGate(s_norm_x, s_norm_y, MUL, bitlen, nvals, no_status);
	share *s_cos_sim = bc->PutFPGate(s_x_dot_y, s_norm_x_times_norm_y, DIV, bitlen, nvals, no_status);
	share *s_cos_sim_out = bc->PutOUTGate(s_cos_sim, SERVER);


	party->ExecCircuit();

	std::cout << "INPUT EMBEDDINGS:" << std::endl;

	std::cout << "X:" << std::endl;

	for (int i = 0; i < nvals; i++) {
		std::cout << *(double *)&xvals[i] << ", ";
	}

	std::cout << std::endl;

	std::cout << "Y:" << std::endl;

	for (int i = 0; i < nvals; i++) {
		std::cout << *(double *)&yvals[i] << ", ";
	}


	std::cout << std::endl;


	std::cout << "VERIFICATION:" << std::endl;
	
	std::cout << "x dot y: " << ver_x_dot_y << std::endl;

	std::cout << "norm(x): " << ver_norm_x << std::endl;

	std::cout << "norm(y): " << ver_norm_y << std::endl;

	std::cout << "cos dist: " << ver_cos_dist << std::endl;

	std::cout << "CIRCUIT RESULTS:" << std::endl;


	uint32_t *cos_sim_out_vals = (uint32_t *)s_cos_sim_out->get_clear_value_ptr();
	double cos_sim = *((double *)cos_sim_out_vals);

	uint32_t *x_dot_y_out_vals = (uint32_t *)s_x_dot_y_out->get_clear_value_ptr();
	double x_dot_y = *((double *)x_dot_y_out_vals);

	uint32_t *norm_x_out_vals = (uint32_t *)s_norm_x_out->get_clear_value_ptr();
	double norm_x = *((double *)norm_x_out_vals);

	uint32_t *norm_y_out_vals = (uint32_t *)s_norm_y_out->get_clear_value_ptr();
	double norm_y = *((double *)norm_y_out_vals);

	std::cout << "x dot y: " << x_dot_y << std::endl;
	std::cout << "norm(x) : " << norm_x << std::endl;
	std::cout << "norm(y): " << norm_y << std::endl;
	std::cout << "cos dist: " << 1 - cos_sim << std::endl;
	
}

int main(int argc, char **argv)
{

	e_role role;
	uint32_t bitlen = 64, nvals = 128, secparam = 128, nthreads = 1;

	uint16_t port = 7766;
	std::string address = "127.0.0.1";
	int32_t test_op = -1;
	e_mt_gen_alg mt_alg = MT_OT;
	uint32_t test_bit = 0;
	double fpa = 10.52, fpb = 1.30;
	std::string inputfile;
	std::string outputfile;

	read_test_options(&argc, &argv, &role, &bitlen, &nvals, &secparam, &address,
					  &port, &test_op, &test_bit, &fpa, &fpb, &inputfile, &outputfile);

	std::cout << std::fixed << std::setprecision(10);
	seclvl seclvl = get_sec_lvl(secparam);

	test_verilog_add64_SIMD(role, address, port, seclvl, nvals, nthreads, mt_alg, S_BOOL, fpa, fpb, inputfile, outputfile);

	return 0;
}
