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
					   uint16_t *port, int32_t *test_op, uint32_t *test_bit, e_mt_gen_alg *mt_alg, uint32_t *nr_of_runs, std::string *inputfile, std::string *pffrocd_path)
{

	uint32_t int_role = 0, int_port = 0, int_testbit = 0, int_mt_alg = 0;

	parsing_ctx options[] =
		{{(void *)&int_role, T_NUM, "r", "Role: 0/1", true, false},
		 {(void *)&int_testbit, T_NUM, "i", "test bit", false, false},
		 {(void *)nvals, T_NUM, "n", "Number of parallel operation elements", false, false},
		 {(void *)bitlen, T_NUM, "b", "Bit-length, default 32", false, false},
		 {(void *)secparam, T_NUM, "s", "Symmetric Security Bits, default: 128", false, false},
		 {(void *)address, T_STR, "a", "IP-address, default: localhost", false, false},
		 {(void *)&int_port, T_NUM, "p", "Port, default: 7766", false, false},
		 {(void *)test_op, T_NUM, "t", "Single test (leave out for all operations), default: off", false, false},
		 {(void *)&int_mt_alg, T_NUM, "x", "Arithmetic multiplication triples algorithm", false, false},
		 {(void *)nr_of_runs, T_NUM, "d", "how many runs of the test", true, false},
		 {(void *)inputfile, T_STR, "f", "Input file containing face embeddings", true, false},
		 {(void *)pffrocd_path, T_STR, "o", "absolute path to pffrocd directory", true, false}
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

	if (int_mt_alg == 0) {
		*mt_alg = MT_OT;
	} else if (int_mt_alg == 1) {
		*mt_alg = MT_PAILLIER;
	} else if (int_mt_alg == 2) {
		*mt_alg = MT_DGK;
	} else {
		std::cout << "Invalid MT algorithm" << std::endl;
		exit(EXIT_FAILURE);
	} 
}

void test_verilog_add64_SIMD(e_role role, const std::string &address, uint16_t port, seclvl seclvl, uint32_t nvals, uint32_t nthreads,
							 e_mt_gen_alg mt_alg, e_sharing sharing, uint32_t nr_of_runs, std::string inputfile, std::string pffrocd_path)
{

	// std::cout << "SEC LEVEL: " << seclvl.symbits << std::endl;
	// std::cout << "MT_ALG: " << mt_alg << std::endl;

	// for addition we operate on doubles, so set bitlen to 64 bits
	uint32_t bitlen = 64;

	std::string circuit_dir = pffrocd_path + "/ABY/bin/circ/";

	//std::cout << "CIRCUIT DIRECTORY: " << circuit_dir << std::endl;

	ABYParty *party = new ABYParty(role, address, port, seclvl, bitlen, nthreads, mt_alg, 100000, circuit_dir);

	// std::cout << "party created" << std::endl;


	std::vector<Sharing *> &sharings = party->GetSharings();

	// std::cout << "circuit retrieved" << std::endl;
	// std::cout << "here 1" << std::endl;

	

	for (int i = 0; i < nr_of_runs; i++) {

		// two arrays of real-world embeddings
		std::vector<double> xembeddings;
		std::vector<double> yembeddings;

		// array for the Sy<role> share
		std::vector<double> share_embeddings;


		// reading the non-xored embeddings, i.e. current face and database face

		std::string file_name = inputfile + std::to_string(i) + ".txt";

		std::fstream infile(file_name);

		//std::cout << "INPUT FILE NAME: " << file_name << std::endl;

		double x, y;

		//std::cout << "starting reading x and y" << std::endl;

		while (infile >> x >> y) {
			// std::cout << "x: " << x << " | y: "<< y << std::endl;
			xembeddings.push_back(x);
			yembeddings.push_back(y);
		}

		//std::cout<<"finished reading x and y" << std::endl;

		assert(xembeddings.size() == nvals);
		assert(yembeddings.size() == nvals);

		// reading the xored embedding, i.e. either Sy<0> or Sy<1> depending on the role

		// char *fname = (char *) malloc(150); // file name buffer 
		// sprintf(fname, "/home/dietpi/pffrocd/ABY/build/bin/share%d.txt", role);

		std::string fname = pffrocd_path + "/ABY/build/bin/share" + std::to_string(role) + ".txt";

		//std::cout << "FNAME: " << fname << std::endl; 

		std::fstream infile_share(fname);

		double z;

		// std::cout << "starting reading z" << std::endl;

		while(infile_share >> z) {
			//std::cout << "z: " << z << std::endl;
			share_embeddings.push_back(z);
		}

		//std::cout<<"finished reading z" << std::endl;

		assert(share_embeddings.size() == nvals);

		// arrays of integer pointers to doubles
		uint64_t xvals[nvals];
		uint64_t yvals[nvals];
		uint64_t sharevals[nvals];
		// std::cout << "here 1" << std::endl;

		// verification in plaintext
		double ver_x_times_y[nvals];
		double ver_x_times_x[nvals];
		double ver_y_times_y[nvals];
		double ver_x_dot_y = 0;
		double ver_norm_x = 0;
		double ver_norm_y = 0;
		// std::cout << "here 1" << std::endl;

		// S_c(X,Y) = (X \dot Y) / (norm(X) * norm(Y))



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
		}

		// std::cout << "here 1" << std::endl;

		ver_norm_x = sqrt(ver_norm_x);
		ver_norm_y = sqrt(ver_norm_y);

		double ver_cos_sim = 1 - (ver_x_dot_y / (ver_norm_x * ver_norm_y));

		// std::cout << "verification values computed" << std::endl;

		// shr_server_set[0] = bc->PutSIMDINGate(nvals, xvals, bitlen, SERVER);
		// shr_client_set[0] = bc->PutSIMDINGate(nvals, yvals, bitlen, CLIENT);

		// OLD INPUT
		// share *s_xin = bc->PutSIMDINGate(nvals, xvals, bitlen, SERVER);
		// share *s_yin = bc->PutSIMDINGate(nvals, yvals, bitlen, CLIENT);


		BooleanCircuit *bc = (BooleanCircuit *)sharings[S_BOOL]->GetCircuitBuildRoutine();

		// INPUTS
		share *s_xin, *s_yin;

		// Input of the current face captured by the drone

		// std::cout << "assigning ingates" << std::endl;

		if(role == SERVER) {
			s_xin = bc->PutSIMDINGate(nvals, xvals, bitlen, SERVER);
		} else { //role == CLIENT
			s_xin = bc->PutDummySIMDINGate(nvals, bitlen);
		}

		// Input of the pre-computed shares of the face in the database

		s_yin = bc->PutSharedSIMDINGate(nvals, sharevals, bitlen);


		share *s_x_times_y = bc->PutFPGate(s_xin, s_yin, MUL, bitlen, nvals, no_status);


		// computing x \dot y
		uint32_t posids[3] = {0, 0, 1};
		// share *s_product_first_wire = s_product->get_wire_ids_as_share(0);
		share *s_x_dot_y = bc->PutSubsetGate(s_x_times_y, posids, 1, true);
		for (int i = 1; i < nvals; i++)
		{
			//uint32_t posids[3] = {i, i, 1};

				posids[0] = i;
				posids[1] = i;
				posids[2] = 1;

			// bc->PutPrintValueGate(bc->PutSubsetGate(s_product,posids,1,false), "First wire");

			// share *s_product_split;
			s_x_dot_y = bc->PutFPGate(s_x_dot_y , bc->PutSubsetGate(s_x_times_y,posids,1,true),ADD);
			//std::cout << "s_share nvals: " << a_share->get_nvals() << std::endl;
			//std::cout << "s_share bitlen: " << a_share->get_bitlength() << std::endl;
			//bc->PutPrintValueGate(s_x_dot_y, "s_x_dot_y");
		}

		// std::cout << "finished computing x dot y" << std::endl;
		//bc->PutPrintValueGate(s_x_dot_y, "s_x_dot_y");
		// share *s_x_dot_y_out = bc->PutOUTGate(s_x_dot_y, SERVER);


		// computing norm(X)

		share *s_x_times_x = bc->PutFPGate(s_xin, s_xin, MUL, bitlen, nvals, no_status);
		// bc->PutPrintValueGate(s_x_times_x, "s_x_times_x");

		// std::cout << "multiplied" << std::endl;


		posids[0] = 0;
		posids[1] = 0;
		posids[2] = 1;
		// share *s_product_first_wire = s_product->get_wire_ids_as_share(0);
		share *s_norm_x = bc->PutSubsetGate(s_x_times_x, posids, 1, true);
		for (int i = 1; i < nvals; i++)
		{
			//uint32_t posids[3] = {i, i, 1};

				posids[0] = i;
				posids[1] = i;
				posids[2] = 1;

			// bc->PutPrintValueGate(bc->PutSubsetGate(s_product,posids,1,false), "First wire");

			// share *s_product_split;
			s_norm_x = bc->PutFPGate(s_norm_x , bc->PutSubsetGate(s_x_times_x,posids,1,true),ADD);
			//std::cout << "s_share nvals: " << a_share->get_nvals() << std::endl;
			//std::cout << "s_share bitlen: " << a_share->get_bitlength() << std::endl;
			//bc->PutPrintValueGate(a_share, "a_share");
		}

		s_norm_x = bc->PutFPGate(s_norm_x, SQRT);

		// std::cout << "finished computing norm(x)" << std::endl;
		// bc->PutPrintValueGate(s_norm_x, "s_norm_x");

		// share *s_norm_x_out = bc->PutOUTGate(s_norm_x, SERVER);


		// computing norm(Y)

		share *s_y_times_y = bc->PutFPGate(s_yin, s_yin, MUL, bitlen, nvals, no_status);
		// bc->PutPrintValueGate(s_y_times_y, "s_y_times_y");

		// std::cout << "multiplied" << std::endl;


		posids[0] = 0;
		posids[1] = 0;
		posids[2] = 1;
		// share *s_product_first_wire = s_product->get_wire_ids_as_share(0);
		share *s_norm_y = bc->PutSubsetGate(s_y_times_y, posids, 1, true);
		for (int i = 1; i < nvals; i++)
		{
			//uint32_t posids[3] = {i, i, 1};

				posids[0] = i;
				posids[1] = i;
				posids[2] = 1;

			// bc->PutPrintValueGate(bc->PutSubsetGate(s_product,posids,1,false), "First wire");

			// share *s_product_split;
			s_norm_y = bc->PutFPGate(s_norm_y , bc->PutSubsetGate(s_y_times_y,posids,1,true),ADD);
			//std::cout << "s_share nvals: " << a_share->get_nvals() << std::endl;
			//std::cout << "s_share bitlen: " << a_share->get_bitlength() << std::endl;
			//bc->PutPrintValueGate(a_share, "a_share");
		}

		s_norm_y = bc->PutFPGate(s_norm_y, SQRT);
		// bc->PutPrintValueGate(s_norm_y, "s_norm_y");

		// std::cout << "finished computing norm(y)" << std::endl;

		// share *s_norm_y_out = bc->PutOUTGate(s_norm_y, SERVER);


		share *s_norm_x_times_norm_y = bc->PutFPGate(s_norm_x, s_norm_y, MUL);
		// bc->PutPrintValueGate(s_norm_x_times_norm_y, "s_norm_x_times_norm_y");

		share *s_cos_sim = bc->PutFPGate(s_x_dot_y, s_norm_x_times_norm_y, DIV);
		// bc->PutPrintValueGate(s_cos_sim, "s_cos_sim");

		share *s_cos_sim_out = bc->PutOUTGate(s_cos_sim, ALL);

		// std::cout << "finished computing cos dist" << std::endl;


		party->ExecCircuit();


		uint32_t out_bitlen_x_times_y, out_nvals_x_times_y;
		uint64_t *out_vals_x_times_y;

		std::cout << std::endl << "cos_dist_ver: " << ver_cos_sim << std::endl;


		uint32_t *cos_sim_out_vals = (uint32_t *)s_cos_sim_out->get_clear_value_ptr();
		double cos_sim = *((double *)cos_sim_out_vals);
		std::cout << "cos_dist: " << 1 - cos_sim << std::endl;

		party -> Reset();

		std::cout << "split here" <<std::endl;

	}

	
	
	// // retrieve plantext output
	// uint32_t out_bitlen_product, out_nvals;
	// uint64_t *out_vals_product;

	// s_product_out->get_clear_value_vec(&out_vals_product, &out_bitlen_product, &out_nvals);

	// printing result

	// std::cout << "INPUT EMBEDDINGS:" << std::endl;

	// std::cout << "X:" << std::endl;

	// for (int i = 0; i < nvals; i++) {
	// 	std::cout << *(double *)&xvals[i] << ", ";
	// }

	// std::cout << std::endl;

	// std::cout << "Y:" << std::endl;

	// for (int i = 0; i < nvals; i++) {
	// 	std::cout << *(double *)&yvals[i] << ", ";
	// }


	// std::cout << std::endl;

	// std::cout << "SHARE:" << std::endl;

	// for (int i = 0; i < nvals; i++) {
	// 	std::cout << *(double *)&sharevals[i] << ", ";
	// }

	// std::cout << std::endl;


	// std::cout << "PARTIAL RESULTS:" << std::endl;

	// std::cout << "<variable> = ver_result --- circ_result" << std::endl;



	//s_x_times_y_out->get_clear_value_vec(&out_vals_x_times_y, &out_bitlen_x_times_y, &out_nvals_x_times_y);

	// for (uint32_t i = 0; i < nvals; i++)
	// {
	// 	double x_times_y_i = *((double *)&out_vals_x_times_y[i]);
	// 	std::cout << "x_times_y[" << i << "] = " << ver_x_times_y[i] << " --- "<< x_times_y_i << std::endl;
	// }

	// std::cout << std::endl;


	// std::cout << "VERIFICATION:" << std::endl;
	
	// std::cout << "x dot y: " << ver_x_dot_y << std::endl;

	// std::cout << "norm(x): " << ver_norm_x << std::endl;

	// std::cout << "norm(y): " << ver_norm_y << std::endl;


	// std::cout << "CIRCUIT RESULTS:" << std::endl;

	// std::cout << "s_product nvals: " << s_product->get_nvals() << std::endl;
	// std::cout << "s_product bitlength: " << s_product->get_bitlength() << std::endl;
	// double sum = 0;

	// for (uint32_t i = 0; i < nvals; i++)
	// {
	// 	// dereference output value as double without casting the content
	// 	double val = *((double *)&out_vals_product[i]);
	// 	double orig_x_i = *(double *)&xvals[i];
	// 	double orig_y_i = *(double *)&yvals[i];
	// 	double ver_result = (orig_x_i * orig_y_i);
	// 	sum = sum + ver_result;
	// 	std::cout << i << " | circuit product: " << val << " --- "
	// 			  << "verification: " << ver_result << " = " << *(double *)&xvals[i] << " * " << *(double *)&yvals[i] << std::endl;
	// 	std::cout << "SUM: " << sum	 << std::endl;
	// }



	// uint32_t *x_dot_y_out_vals = (uint32_t *)s_x_dot_y_out->get_clear_value_ptr();
	// double x_dot_y = *((double *)x_dot_y_out_vals);

	// uint32_t *norm_x_out_vals = (uint32_t *)s_norm_x_out->get_clear_value_ptr();
	// double norm_x = *((double *)norm_x_out_vals);

	// uint32_t *norm_y_out_vals = (uint32_t *)s_norm_y_out->get_clear_value_ptr();
	// double norm_y = *((double *)norm_y_out_vals);

	// std::cout << "x dot share: " << x_dot_y << std::endl;
	// std::cout << "norm(x) : " << norm_x << std::endl;
	// std::cout << "norm(share): " << norm_y << std::endl;

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
	uint32_t nr_of_runs = 1;
	std::string inputfile;
	std::string pffrocd_path;

	read_test_options(&argc, &argv, &role, &bitlen, &nvals, &secparam, &address,
					  &port, &test_op, &test_bit, &mt_alg, &nr_of_runs, &inputfile, &pffrocd_path);

	std::cout << std::fixed << std::setprecision(10);
	seclvl seclvl = get_sec_lvl(secparam);

	test_verilog_add64_SIMD(role, address, port, seclvl, nvals, nthreads, mt_alg, S_BOOL, nr_of_runs, inputfile, pffrocd_path);

	return 0;
}
