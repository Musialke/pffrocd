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

void read_test_options(int32_t *argcp, char ***argvp, e_role *role,
					   uint32_t *bitlen, uint32_t *nvals, uint32_t *secparam, std::string *address,
					   uint16_t *port, int32_t *test_op, uint32_t *test_bit, double *fpa, double *fpb)
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
		 {(void *)fpb, T_DOUBLE, "y", "FP b", false, false}

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
							 e_mt_gen_alg mt_alg, e_sharing sharing, double afp, double bfp)
{

	// for addition we operate on doubles, so set bitlen to 64 bits
	uint32_t bitlen = 64;

	std::string circuit_dir = "../../bin/circ/";

	ABYParty *party = new ABYParty(role, address, port, seclvl, bitlen, nthreads, mt_alg, 100000, circuit_dir);

	std::vector<Sharing *> &sharings = party->GetSharings();

	BooleanCircuit *bc = (BooleanCircuit *)sharings[S_BOOL]->GetCircuitBuildRoutine();
	ArithmeticCircuit *ac = (ArithmeticCircuit *)sharings[S_ARITH]->GetCircuitBuildRoutine();
	Circuit *yc = (BooleanCircuit *)sharings[S_YAO]->GetCircuitBuildRoutine();

	// two arrays of real-world embeddings
	double xembeddings[nvals] = {-0.7139217257499695, 0.779105544090271, 1.03533935546875, 0.19113388657569885, -0.8952100276947021, -0.5631617307662964, 0.12302472442388535, -0.4038945436477661, 1.0790332555770874, -0.39052635431289673, -0.8452416658401489, 1.0576269626617432, 0.7670206427574158, 1.1732330322265625, 0.2067546248435974, 0.5163469910621643, -0.37744832038879395, -1.1916933059692383, -0.10566246509552002, 1.1311050653457642, 0.3539687991142273, -0.04446204751729965, 0.3907996714115143, -0.6490335464477539, -0.5599715709686279, -0.2612435519695282, -1.9377541542053223, 0.011267831549048424, 0.4115447998046875, 0.2407277226448059, 1.63446843624115, 1.0264508724212646, 1.0686869621276855, 0.4350345730781555, 1.1628575325012207, -2.2198731899261475, 0.023936569690704346, -0.9671810865402222, -0.776322603225708, -0.2108483910560608, -0.7473869323730469, -0.7857978343963623, 0.9489213228225708, 0.6946954727172852, 0.6303039789199829, 0.2564682364463806, 0.9804896116256714, -0.5837778449058533, 0.9779276847839355, -0.5229518413543701, -0.42070192098617554, 0.11629107594490051, -0.1946457177400589, -0.6514270901679993, -1.0945168733596802, 0.12886586785316467, -0.3724145293235779, 0.4713052213191986, 0.5164777040481567, -1.6852772235870361, -0.7984651327133179, 0.7699068188667297, -0.3623180687427521, 0.6481657028198242, 0.9130766987800598, -0.2871786952018738, -0.6433311104774475, -0.7074404954910278, 0.24564208090305328, -0.336145281791687, 1.3128688335418701, 1.3216640949249268, 0.26922744512557983, 0.6511465907096863, 0.1672307699918747, 1.5610928535461426, -1.9497450590133667, 0.08045151829719543, -0.2961958050727844, -0.03209330886602402, -0.01657203957438469, 0.11477973312139511, -0.19689878821372986, 1.3522948026657104, -0.7230944633483887, 0.5718117356300354, 1.2671376466751099, -1.8062065839767456, -1.6252652406692505, -1.5481833219528198, -0.49607226252555847, 0.6559416055679321, 0.19399486482143402, -0.10946081578731537, 0.9284181594848633, 0.30447253584861755, 0.1909874826669693, 1.1636898517608643, -1.656119704246521, 0.74617600440979, 0.03511644899845123, 0.8254475593566895, 2.132702350616455, -0.7526997327804565, -0.2070029079914093, -1.307081699371338, -0.2387426495552063, -0.7231435179710388, 0.18293185532093048, -0.21718524396419525, 1.2594213485717773, 0.13336968421936035, -0.1627960503101349, 1.397972583770752, -0.5820574760437012, 1.2329318523406982, 0.0182635560631752, 0.32219475507736206, -1.1142810583114624, 1.0486818552017212, 1.1416651010513306, 0.7766472697257996, -0.4313766360282898, 1.3483251333236694, -0.992262601852417, 0.4936714172363281, 0.18162153661251068, 0.019330209121108055}; 
	double yembeddings[nvals] = {-0.2669537365436554, 0.6284178495407104, 0.15322162210941315, 3.385287284851074, -0.8913129568099976, 1.1622477769851685, -1.3101105690002441, 0.23956605792045593, -0.09989598393440247, 0.08595024794340134, 0.6507077217102051, 0.6974311470985413, 1.8171982765197754, 2.5553791522979736, 2.0395560264587402, 0.9396897554397583, 1.359078288078308, -2.1641597747802734, 0.46566200256347656, 1.4603056907653809, 1.4958117008209229, -1.3511852025985718, 0.9363498687744141, 1.2203158140182495, -0.7891054749488831, -1.2380822896957397, -0.961654543876648, -0.9877721667289734, -0.2095857560634613, 0.49677973985671997, 0.6621701717376709, -0.16647115349769592, 0.7873673439025879, -0.2825338840484619, 1.2179280519485474, 0.6472935080528259, -0.3485824167728424, -1.7306089401245117, 0.03221255540847778, 0.5856603980064392, -1.2390449047088623, -1.4299955368041992, -0.5266529321670532, -0.8828799724578857, -0.4218916594982147, -1.5213698148727417, 0.8087632060050964, -1.3383797407150269, 0.3999564051628113, 1.2641953229904175, -1.6707204580307007, 0.7092372179031372, -0.9246513843536377, 0.9387121796607971, -0.3805389404296875, 0.9970550537109375, -1.16483473777771, 0.6123709678649902, -0.3597378134727478, -0.959926426410675, -0.18019098043441772, 1.521348476409912, -1.6174430847167969, 1.3873895406723022, 1.0122863054275513, 0.22457057237625122, -2.1137020587921143, -0.28370311856269836, -0.05438928306102753, 0.27258217334747314, 0.886117160320282, 0.8216903209686279, -0.33665066957473755, -0.1711321473121643, -0.3783551752567291, 0.37482255697250366, -2.5529441833496094, 0.19435928761959076, -0.08208899199962616, 0.17379605770111084, 0.7564132809638977, 1.183481216430664, 1.2145601511001587, 0.08859732747077942, -0.26574984192848206, 1.609129786491394, -0.40914565324783325, -1.6024869680404663, -1.891158938407898, -0.3034048080444336, -0.15902414917945862, 0.8432437777519226, -0.5248093605041504, -0.5706746578216553, -0.7086155414581299, 0.9934972524642944, 0.16499952971935272, 0.7436102628707886, -1.649275779724121, 0.8742146492004395, -1.327034592628479, -1.8740352392196655, 0.3169921636581421, -1.128474235534668, -0.7107204794883728, 0.27445244789123535, 0.15215492248535156, -0.3267475962638855, 0.44381871819496155, -1.9094854593276978, 2.14876651763916, 0.5151187181472778, -0.3889283239841461, 1.392277717590332, -0.04372502863407135, 3.005310297012329, 0.5809190273284912, -1.4070483446121216, 0.7667227387428284, -0.5219079256057739, 0.1343550831079483, 0.6196316480636597, 0.6263742446899414, -0.5325893759727478, 0.4179970622062683, -0.4176422655582428, -1.0421963930130005, 0.028064822778105736};

	// two arrays of random doubles
	uint64_t xvals[nvals];
	uint64_t yvals[nvals];


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
		double random_x = xembeddings[i];
		double random_y = yembeddings[i];

		uint64_t *xptr = (uint64_t *)&random_x;
		uint64_t *yptr = (uint64_t *)&random_y;

		xvals[i] = *xptr;
		yvals[i] = *yptr;

		ver_x_times_y[i] = random_x * random_y;
		ver_x_dot_y += ver_x_times_y[i];

		ver_x_times_x[i] = random_x * random_x;
		ver_y_times_y[i] = random_y * random_y;
		ver_norm_x += ver_x_times_x[i];
		ver_norm_y += ver_y_times_y[i];
 

		// shr_server_set[i] = yc->PutSIMDINGate(bitlen, xvals[i], 1, SERVER);
		// shr_client_set[i] = yc->PutSIMDINGate(bitlen, yvals[i], 1, CLIENT);
	}

	ver_norm_x = sqrt(ver_norm_x);
	ver_norm_y = sqrt(ver_norm_y);

	double ver_cos_sim = ver_x_dot_y / (ver_norm_x * ver_norm_y);

	// shr_server_set[0] = bc->PutSIMDINGate(nvals, xvals, bitlen, SERVER);
	// shr_client_set[0] = bc->PutSIMDINGate(nvals, yvals, bitlen, CLIENT);

	// SIMD input gates
	share *s_xin = bc->PutSIMDINGate(nvals, xvals, bitlen, SERVER);
	share *s_yin = bc->PutSIMDINGate(nvals, yvals, bitlen, CLIENT);

	// for (uint32_t i = 0; i<nvals;i++){
	// 	shr_out[i] = bc->PutFPGate(shr_server_set[i], shr_client_set[i], MUL, 1, bitlen, no_status);
	// 	std::cout << "s_product_split nvals: " << shr_out[i]->get_nvals() << std::endl;
	// 	std::cout << "s_product_splitbitlen: " << shr_out[i]->get_bitlength() << std::endl;
	// 	shr_out[i] = bc->PutSplitterGate(shr_out[i]);
	// 	std::cout << "s_product_split nvals: " << shr_out[i]->get_nvals() << std::endl;
	// 	std::cout << "s_product_splitbitlen: " << shr_out[i]->get_bitlength() << std::endl;
	// }

	share *s_x_times_y = bc->PutFPGate(s_xin, s_yin, MUL, bitlen, nvals, no_status);
	bc->PutPrintValueGate(s_x_times_y, "s_x_times_y");
	share *s_x_times_y_out = bc->PutOUTGate(s_x_times_y, ALL);

	// std::cout << "wire(0) " <<s_product->get_nvals_on_wire(0) << std::endl;

	//share *s_product_split = bc->PutCombinerGate(bc->PutSplitterGate(s_product));

	std::cout << "s_product nvals: " << s_x_times_y->get_nvals() << std::endl;
	std::cout << "s_product bitlen: " << s_x_times_y->get_bitlength() << std::endl;

	//bc->PutPrintValueGate(s_product_split, "s_product_split");
	//bc->PutPrintValueGate(s_product, "s_product");

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
	//bc->PutPrintValueGate(s_x_dot_y, "s_x_dot_y");
	share *s_x_dot_y_out = bc->PutOUTGate(s_x_dot_y, ALL);


	// computing norm(X)

	share *s_x_times_x = bc->PutFPGate(s_xin, s_xin, MUL, bitlen, nvals, no_status);
	bc->PutPrintValueGate(s_x_times_x, "s_x_times_x");


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
	bc->PutPrintValueGate(s_norm_x, "s_norm_x");

	share *s_norm_x_out = bc->PutOUTGate(s_norm_x, ALL);


	// computing norm(Y)

	share *s_y_times_y = bc->PutFPGate(s_yin, s_yin, MUL, bitlen, nvals, no_status);
	bc->PutPrintValueGate(s_y_times_y, "s_y_times_y");


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
	bc->PutPrintValueGate(s_norm_y, "s_norm_y");

	share *s_norm_y_out = bc->PutOUTGate(s_norm_y, ALL);


	share *s_norm_x_times_norm_y = bc->PutFPGate(s_norm_x, s_norm_y, MUL);
	bc->PutPrintValueGate(s_norm_x_times_norm_y, "s_norm_x_times_norm_y");

	share *s_cos_sim = bc->PutFPGate(s_x_dot_y, s_norm_x_times_norm_y, DIV);
	bc->PutPrintValueGate(s_cos_sim, "s_cos_sim");


	share *s_cos_sim_out = bc->PutOUTGate(s_cos_sim, ALL);

	// for (int i = 1; i<2; i++) {
	// 	posids[0] = i;
	// 	posids[1] = i;
	// 	posids[2] = 1;
	// 	a_share = bc->PutFPGate(a_share , bc->PutSubsetGate(s_product,posids,1,false),ADD,bitlen,1,no_status);
	// 	a_share = s_product->get_wire_ids_as_share(i);
	// }
	// std::cout << "a share nvals: " << a_share->get_nvals() << std::endl;
	// std::cout << "a share bitlen: " << a_share->get_bitlength() << std::endl;
	// std::cout << "s_product_first_wire share nvals: " << s_product_first_wire->get_nvals() << std::endl;
	// std::cout << "s_product_first_wire share bitlen: " << s_product_first_wire->get_bitlength() << std::endl;
	// std::cout << "here1" << std::endl;
	// s_product_first_wire = bc->PutFPGate(s_product_first_wire, a_share, ADD, s_product_split->get_bitlength(), s_product_split->get_nvals(), no_status);
	// s_product_split->set_wire_id(0, bc->PutADDGate(s_product_split->get_wire_ids_as_share(0), s_product_split->get_wire_ids_as_share(i)));
	// s_product_first_wire = bc->PutADDGate(s_product_first_wire, s_product_split->get_wire_ids_as_share(i));
	// std::cout << "here2" << std::endl;
	// bc->PutPrintValueGate(s_next_wire, "next wire");

	// share* s_product_out = bc->PutOUTGate(s_product_added, ALL);

	//share *s_product_out = bc->PutOUTGate(a_share, ALL);

	// // testing

	// share *s_x;

	// s_x = ac->PutB2AGate(s_product);

	// // split SIMD gate to separate wires (size many)
	// s_x = ac->PutSplitterGate(s_x);

	// // add up the individual multiplication results and store result on wire 0
	// // in arithmetic sharing ADD is for free, and does not add circuit depth, thus simple sequential adding
	// for (int i = 1; i < nvals; i++) {
	// 	s_x->set_wire_id(0, ac->PutADDGate(s_x->get_wire_id(0), s_x->get_wire_id(i)));
	//}

	// // discard all wires, except the addition result
	// s_x->set_bitlength(1);

	// share *s_product_out = ac->PutOUTGate(s_x, ALL);

	// share *s_product_test = bc->PutSplitterGate(s_product);
	// share *s_temp = bc->PutFPGate(s_product_test->get_wire_ids_as_share(0), s_product_test->get_wire_ids_as_share(1), ADD, bitlen, nvals, no_status);
	// bc->PutPrintValueGate (s_temp,"Temp share\n");
	// for (int i = 1; i < nvals; i++) {
	// share *s_temp = circ->PutFPGate(s_product_test->get_wire_ids_as_share(0), s_product_test->get_wire_ids_as_share(i), ADD, bitlen, nvals, no_status);
	// circ->PutPrintValueGate (s_temp,"Temp share\n");
	// // 	//s_product_test->set_wire_id(0, );
	// }

	party->ExecCircuit();

	// // retrieve plantext output
	// uint32_t out_bitlen_product, out_nvals;
	// uint64_t *out_vals_product;

	// s_product_out->get_clear_value_vec(&out_vals_product, &out_bitlen_product, &out_nvals);

	// printing result

	std::cout << "RANDOM INPUT:" << std::endl;

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

	// --- printing vectors for verification

	std::cout << "PARTIAL RESULTS:" << std::endl;

	std::cout << "<variable> = ver_result --- circ_result" << std::endl;


	uint32_t out_bitlen_x_times_y, out_nvals_x_times_y;
	uint64_t *out_vals_x_times_y;

	s_x_times_y_out->get_clear_value_vec(&out_vals_x_times_y, &out_bitlen_x_times_y, &out_nvals_x_times_y);

	for (uint32_t i = 0; i < nvals; i++)
	{
		double x_times_y_i = *((double *)&out_vals_x_times_y[i]);
		std::cout << "x_times_y[" << i << "] = " << ver_x_times_y[i] << " --- "<< x_times_y_i << std::endl;
	}

	std::cout << std::endl;


	std::cout << "VERIFICATION:" << std::endl;
	
	std::cout << "x dot y: " << ver_x_dot_y << std::endl;

	std::cout << "norm(x): " << ver_norm_x << std::endl;

	std::cout << "norm(y): " << ver_norm_y << std::endl;

	std::cout << "cos sim: " << ver_cos_sim << std::endl;

	std::cout << "CIRCUIT RESULTS:" << std::endl;

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
	std::cout << "cos sim: " << cos_sim << std::endl;

	

	
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

	read_test_options(&argc, &argv, &role, &bitlen, &nvals, &secparam, &address,
					  &port, &test_op, &test_bit, &fpa, &fpb);

	std::cout << std::fixed << std::setprecision(10);
	seclvl seclvl = get_sec_lvl(secparam);

	test_verilog_add64_SIMD(role, address, port, seclvl, nvals, nthreads, mt_alg, S_BOOL, fpa, fpb);

	return 0;
}
