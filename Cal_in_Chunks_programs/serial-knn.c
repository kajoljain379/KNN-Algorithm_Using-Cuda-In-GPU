#include < stdio.h > 
#include < time.h > 
#include < stdlib.h > 
#include < math.h > 
#include < cuda.h > 
#include < driver_types.h > 
#define row 135000
#define col 10
#define test_row 13000
#define test_col 10

void main() {
        clock_t s_time, e_time;
        double t_time;
        FILE * myfile, * myfilet;
        int i, j, k1;
        double train[row][col], test1[test_row][test_col];
        int k;
        printf("Enter the k value to apply k nearest neighbour algorithm");
        scanf("%d", & k);
        int result[test_row];

        int set;
        printf("Enter the total classes present in your dataset\n");
        scanf("%d", & set);
        set = set + 1;
        int class[set];

        printf("\n");
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

        //scanning train data
        for (i = 0; i < row; i++) {
                for (j = 0; j < col; j++) {
                        fscanf(myfile, "%lf", & train[i][j]);
                }
        }

        //scanning test data

        for (i = 0; i < test_row; i++) {
                for (j = 0; j < test_col; j++) {
                        fscanf(myfilet, "%lf", & test1[i][j]);
                }
        }

        for (i = 0; i < row; i++) {
                printf("Point no for train data %d ", i + 1);
                for (j = 0; j < col; j++) {
                        printf("%0.0lf  ", train[i][j]);
                }
                printf("\n");
        }
        //Printing test data

        for (i = 0; i < test_row; i++) {
                printf("Point no for test data %d ", i + 1);
                for (j = 0; j < test_col; j++) {
                        printf("%0.0lf  ", test1[i][j]);
                }
                printf("\n");
        }

        //Here we calculate Euclidian distance
        //Timer start
        s_time = clock();
        for (k1 = 0; k1 < test_row; k1++) {
                float sum = 0;
                double distance[row][2];
                for (i = 0; i < set; i++)
                        class[i] = 0;
                for (i = 0; i < row; i++) {
                        sum = 0;
                        for (j = 0; j < col - 1; j++) {
                                sum += (train[i][j] - test1[k1][j]) * (train[i][j] - test1[k1][j]);
                        }
                        distance[i][0] = sqrt(sum);
                        distance[i][1] = train[i][col - 1];
                }

                // for finding k nearest neighbours

                for (i = 0; i < k; i++) {
                        int min = i;
                        for (j = i + 1; j < row; j++) {
                                if (distance[j][0] < distance[min][0])
                                        min = j;
                        }
                        float temp = distance[i][0];
                        distance[i][0] = distance[min][0];
                        distance[min][0] = temp;
                        temp = distance[i][1];
                        distance[i][1] = distance[min][1];
                        distance[min][1] = temp;

                }

                for (i = 0; i < k; i++) {

                        class[(int) distance[i][1]] += 1;
                }

                //Here we find out in which class that test element belongs
                int max = 1;
                for (i = 2; i < set; i++) {
                        if (class[i] > class[max])
                                max = i;
                }

                printf("Given Test point %d  belongs to class %d\n ", k1 + 1, max);
                result[k1] = max;

        }
        e_time = clock();
        t_time = ((double)(e_time - s_time)) / 1000000;
        int count = 0;
        for (i = 0; i < test_row; i++) {
                if (test1[i][9] != result[i]) {
                        count++;
                }
        }

        printf("Count unmachted %d", count);

        printf("\n \n Total time taken %0.2lf second", t_time);

}
