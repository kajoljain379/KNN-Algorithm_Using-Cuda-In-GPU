#include < stdio.h > 
#include < cuda.h > 
#include < time.h > 
#include < math.h > 
#define row 50000
#define col 10
#define test_row 13000
#define test_col 10

__global__

void class_classification(int * index_chunks, double * * set, int total_chunks_train, int k, int * d_kneighbours, int set1, int * res_class) {
        //double k_nearest[k][2];
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        int set_i;
        int min = 0;
        //int index_chunks[total_chunks_train];
        for (int f = 0; f < total_chunks_train; f++)
                index_chunks[f] = 0;
        if (i < test_row) {
                for (int l = 0; l < k; l++) {
                        min = 0;
                        //set[0][i*test_row+index_chunks[total_chunks_train]*2]; 
                        for (int j = 1; j < total_chunks_train; j++) {
                                if (set[min][(index_chunks[min] * test_row + i) * 2] > set[j][(index_chunks[j] * test_row + i) * 2])
                                        min = j;
                        }
                        //k_nearest[l][0]=set[min][i*test_row+index_chunks[min]*2];
                        //k_nearest[l][1]=set[min][i*test_row+index_chunks[min]*2+1];
                        set_i = i * set1;
                        d_kneighbours[(int)(set[min][(index_chunks[min] * test_row + i) * 2 + 1]) - 1 + set_i] += 1;
                        index_chunks[min]++;
                }
                set_i = i * set1;
                int max = 0;
                for (int l = 1; l < set1; l++) {
                        if (d_kneighbours[set_i + l] > d_kneighbours[set_i + max])
                                max = l;
                }
                res_class[i] = max + 1;

        }
}

__global__

void KminNeighbourFind(double * distance, int k) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        //int index=i*row*2;

        //int set_i;
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

                }
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

        double * distance;
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

        //scanning test data

        //chunk of test cases 
        s_time = clock();
        for (int test_c = 0; test_c < total_chunks_test; test_c++) {
                printf("\nTest Case chunk no %d is on working state", test_c + 1);

                for (i = 0; i < test_row; i++) {
                        for (j = 0; j < test_col; j++) {
                                fscanf(myfilet, "%lf", & test1[i * test_col + j]);
                        }
                }

                myfile = fopen("shuttle.trn", "r");

                double * * set_train_kneigh = (double * * ) malloc(total_chunks_train * sizeof(double * ));
                int * res_class, * h_class;

                for (int h = 0; h < total_chunks_train; h++)
                        set_train_kneigh[h] = (double * ) malloc(2 * k * test_row * sizeof(double));

                for (int train_c = 0; train_c < total_chunks_train; train_c++) {
                        printf("\nTrain Case chunk no %d is on working state", train_c + 1);

                        //myfile=fopen("shuttle.trn","r");

                        for (i = 0; i < row; i++) {
                                for (j = 0; j < col; j++) {
                                        fscanf(myfile, "%lf", & train[i * col + j]);
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

                        cudaMalloc( & d_train, size);
                        cudaMalloc( & d_test, size1);
                        cudaMalloc( & distance, distance_size);
                        //cudaMalloc(&res_class,class_mem);
                        //copy the data from host to device memory
                        cudaMemcpy(d_train, train, size, cudaMemcpyHostToDevice);
                        cudaMemcpy(d_test, test1, size1, cudaMemcpyHostToDevice);

                        //int threads=test_row*row;
                        dim3 dimgrid((row - 1) / 16 + 1, (test_row - 1) / 16 + 1, 1);
                        dim3 dimblock(16, 16, 1);
                        Euclidian_distance << < dimgrid, dimblock >>> (d_train, d_test, distance);
                        //cudaMemcpy(h_distance,distance,distance_size,cudaMemcpyDeviceToHost);

                        cudaFree(d_train);
                        cudaFree(d_test);

                        KminNeighbourFind << < (test_row - 1) / 16 + 1, 16 >>> (distance, k);
                        //double kdistance[2*k*test_row];
                        cudaMemcpy(set_train_kneigh[train_c], distance, 2 * k * test_row * sizeof(double), cudaMemcpyDeviceToHost);

                }
                //class_classification(int index_chunks[],double **set, int total_chunks_train ,int k,int *d_kneighbours,int set1,int *res_class)
                int * index_chunks;
                //int *res_class,*h_class;
                int * d_kneighbours, * h_kneighbours;
                double * * set_nei;
                size_t neighbour_size = test_row * set * sizeof(int);
                cudaMalloc( & d_kneighbours, neighbour_size);
                size_t class_mem = test_row * sizeof(int);
                h_class = (int * ) malloc(class_mem);
                h_kneighbours = (int * ) malloc(neighbour_size);
                cudaMalloc( & res_class, class_mem);
                cudaMalloc( & index_chunks, total_chunks_train * sizeof(int));
                cudaMalloc( & set_nei, test_row * 2 * k * total_chunks_train * sizeof(double));
                cudaMemcpy(set_nei, set_train_kneigh, test_row * 2 * k * total_chunks_train * sizeof(double), cudaMemcpyHostToDevice);

                class_classification << < (test_row - 1) / 16 + 1, 16 >>> (index_chunks, set_nei, total_chunks_train, k, d_kneighbours, set, res_class);
                cudaMemcpy(h_class, res_class, class_mem, cudaMemcpyDeviceToHost);
                cudaMemcpy(h_kneighbours, d_kneighbours, neighbour_size, cudaMemcpyDeviceToHost);
                for (i = 0; i < test_row; i++) {
                        for (j = 0; j < set; j++) {
                                //printf("class freq of test case %d class no %d value %d\n",i+1,j,h_kneighbours[i*set+j]);
                        }
                }

                //cudaFree(distance1);
                cudaFree(d_kneighbours);
                cudaFree(res_class);
                cudaFree(index_chunks);
                cudaFree(set_nei);
                count = 0;

                free(h_kneighbours);
                //free(set_train_kneigh);

                for (i = 0; i < test_row; i++) {
                        if (test1[i * col + col - 1] != h_class[i])
                                count++;
                        //printf("Given Test point %d  belongs to class %d\n",i+1,h_class[i]);
                }

                cudaFree(distance);
                printf("count unmatched in first %d chunk size is %d\n", count, i);
                //count=0;
        }
        e_time = clock();
        t_time = ((double)(e_time - s_time)) / 1000000;
        printf("Count unmachted %d", count);

        printf("\n \n Total time taken %0.2lf second", t_time);

        return 0;

}
