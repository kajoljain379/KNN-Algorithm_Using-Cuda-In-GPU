#include < stdio.h > 
#include < cuda.h > 
#include < time.h > 
#include < math.h > 
#define row 50000
#define col 10
#define test_row 13000
#define test_col 10

__global__

void KminNeighbourFind(double * distance, int * d_kneighbours, int k, int set, int * res_class) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        //int index=i*row*2;

        int set_i;
        if (i < test_row) {
                for (int i1 = 0; i1 < k; i1++) {
                        int min = 2 * (i1 * test_row + i);
                        for (int j1 = i1 + 1; j1 < row; j1++) {
                                if (distance[2 * (j1 * test_row + i)] < distance[min])
                                        min = 2 * (j1 * test_row + i);
                        }

                        int dist = 2 * (i1 * test_row + i), clas = 2 * (i1 * test_row + i) + 1;
                        double temp = distance[dist];
                        distance[dist] = distance[min];
                        distance[min] = temp;
                        temp = distance[clas];
                        distance[clas] = distance[min + 1];
                        distance[min + 1] = temp;
                        int index = (int) distance[clas] - 1;
                        set_i = i * set;
                        index = index + set_i;
                        d_kneighbours[index] += 1;
                }
                //w=distnace1[2*(0*test_row+i)];
                int max = 0;
                for (int l = 1; l < set; l++) {
                        if (d_kneighbours[set_i + l] > d_kneighbours[set_i + max])
                                max = l;
                }
                res_class[i] = max + 1;

        }

}


__global__

void Euclidian_distance(double * d_train, double * d_test, double * distance) {
        int ro = blockIdx.x * blockDim.x + threadIdx.x;
        int co = blockIdx.y * blockDim.y + threadIdx.y;
        int distanceid = 2 * (ro * test_row + co);

        double sum = 0, diff = 0;
        //checking boundary condition
        if (ro < row && co < test_row) {
                for (int i = 0; i < col - 1; i++) {

                        diff = (d_train[ro * col + i] - d_test[co * col + i]);
                        sum += diff * diff;
                }
                distance[distanceid] = sqrt(sum);
                distance[distanceid + 1] = d_train[ro * col + col - 1];
        }

        // __syncthreads();
}


int main() {
        int count;
        clock_t s_time, e_time;
        double t_time;
        FILE * myfile, * myfilet;
        int k, i, j;
        double train[row * col], test1[test_row * test_col];
        double * d_train, * d_test;

        double * distance, * h_distance, * h_distance1;
        printf("Enter the k value to apply k nearest neighbour algorithm");
        scanf("%d", & k);

        printf("\n");
        int set;
        printf("Enter the total classes present in your dataset\n");
        scanf("%d", & set);

        myfile = fopen("shuttle.trn", "r");
        if (myfile == NULL) {
                printf("data not open\n");
                exit(0);
        } else {
                printf("Successfully open\n");
        }

        myfilet = fopen("shuttle.tst", "r");
        if (myfilet == NULL) {
                printf("Test data not open\n");
                exit(0);
        } else {
                printf("Test file open successfully\n");
        }

        //Test cases and train set

        int total_train, total_test;
        printf("Enter total no of train data elements\n");
        scanf("%d", & total_train);
        printf("Enter total no of test data elements\n");
        scanf("%d", & total_test);

        int total_chunks_train = (total_train - 1) / row + 1;
        int total_chunks_test = (total_test - 1) / test_row + 1;

        printf("Total train and test chunks are %d and %d \n", total_chunks_train, total_chunks_test);
        //scanning train data
        for (i = 0; i < row; i++) {
                for (j = 0; j < col; j++) {
                        fscanf(myfile, "%lf", & train[i * col + j]);
                }
        }

        //scanning test data

        //chunk of test cases 
        s_time = clock();
        for (int test_c = 0; test_c < total_chunks_test; test_c++) {

                for (i = 0; i < test_row; i++) {
                        for (j = 0; j < test_col; j++) {
                                fscanf(myfilet, "%lf", & test1[i * test_col + j]);
                        }
                }

                cudaError_t cudastatus;
                cudastatus = cudaDeviceReset();
                if (cudastatus != cudaSuccess) {
                        fprintf(stderr, " cudaDeviceReset failed!");
                        return 1;
                }
                cudastatus = cudaSetDevice(0);
                if (cudastatus != cudaSuccess) {
                        fprintf(stderr, " cudaSetDevice failed!");
                        return 1;
                } else
                        printf(" Working \n ");

                //s_time=clock();

                size_t size = row * col * sizeof(double);
                size_t size1 = test_row * test_col * sizeof(double);

                size_t distance_size = 2 * row * test_row * sizeof(double);
                size_t class_mem = test_row * sizeof(int);
                //d_train=(double*)malloc(size);
                //d_test=(double*)malloc(size1);
                int * res_class, * h_class;
                h_distance = (double * ) malloc(distance_size);
                //h_distance1=(double*)malloc(distance_size);
                h_class = (int * ) malloc(class_mem);
                //* Allocate matrices in device memory 
                cudaMalloc( & d_train, size);
                cudaMalloc( & d_test, size1);
                cudaMalloc( & distance, distance_size);
                cudaMalloc( & res_class, class_mem);
                //copy the data from host to device memory
                cudaMemcpy(d_train, train, size, cudaMemcpyHostToDevice);
                cudaMemcpy(d_test, test1, size1, cudaMemcpyHostToDevice);

                //int threads=test_row*row;
                dim3 dimgrid((row - 1) / 16 + 1, (test_row - 1) / 16 + 1, 1);
                dim3 dimblock(16, 16, 1);
                Euclidian_distance << < dimgrid, dimblock >>> (d_train, d_test, distance);

                cudaFree(d_train);
                cudaFree(d_test);

                //here code for min k neighbour cal

                size_t neighbour_size = test_row * set * sizeof(int);
                int * d_kneighbours;

                cudaMalloc( & d_kneighbours, neighbour_size);
                //cudaMemcpy(distance,h_distance,distance_size,cudaMemcpyHostToDevice);

                int h_kneighbours[neighbour_size];
                KminNeighbourFind << < (test_row - 1) / 16 + 1, 16 >>> (distance, d_kneighbours, k, set, res_class);
                double kdistance[2 * k * test_row];
                cudaMemcpy(kdistance, distance, 2 * k * test_row * sizeof(double), cudaMemcpyDeviceToHost);

                cudaMemcpy(h_class, res_class, class_mem, cudaMemcpyDeviceToHost);
                //cudaFree(distance1);
                cudaFree(d_kneighbours);
                cudaFree(res_class);

                for (i = 0; i < test_row; i++) {
                        if (test1[i * col + col - 1] != h_class[i])
                                count++;
                        //printf("Given Test point %d  belongs to class %d\n",i+1,h_class[i]);
                }
                cudaFree(distance);
        }

        e_time = clock();
        t_time = ((double)(e_time - s_time)) / 1000000;
        printf("Count unmachted %d", count);

        printf("\n \n Total time taken %0.2lf second", t_time);

        return 0;

}
