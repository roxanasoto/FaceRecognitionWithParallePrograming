#include <iostream>
#include <stdlib.h>
#include <vector>
#include <math.h>
#include <time.h>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <string>
#include <sstream>
#include <cuda_runtime.h>

using namespace std;

#define DATA_SIZE 150 // Numero de datos usados
#define TRAIN_DSIZE 120 // Numero de datos de entrenamiento
#define TEST_DSIZE 30 // Numero de datos de testeo
#define INPUT_SIZE 15 // Numero de neuronas en capa de entrada
#define HIDDEN_LAYERS 1 // Numero de capas ocultas
#define HIDDEN_SIZE 15 // Numero de neuronas en capas ocultas
#define OUTPUT_SIZE 15
#define EPOCHS 200000
#define BATCHSIZE 16
#define N 48

float error;
int topology[HIDDEN_LAYERS+2];
float X_TRAIN[TRAIN_DSIZE*INPUT_SIZE];
float Y_TRAIN[TRAIN_DSIZE*OUTPUT_SIZE];

float X_TEST[TEST_DSIZE*INPUT_SIZE];
float Y_TEST[TEST_DSIZE*OUTPUT_SIZE]={0};

//// Activacion forward
__device__ float sigmoid(float x) {
  return (1.0f / (1.0f + exp(-x)));
}
//// Derivada de activacion backward
__device__ float dsigmoid(float x) {
  return (x*(1-x));
}


__global__ void training(float *dev_W,int *topology,float *X_TRAIN,float *Y_TRAIN) {
  int offset=threadIdx.x+blockIdx.x*blockDim.x;
  float LEARNING_RATE=0.5;
  ///////////////////////////////////////////////////////
                    /*INICIALIZACIONES*/
  //////////////////////////////////////////////////////
  if(offset<N){

    float ***W;
    W = new float**[HIDDEN_LAYERS+1];
    for(int l = 0; l < HIDDEN_LAYERS+1; l++){
      W[l] = new float*[topology[l]+1];//+1 for bias
      for (int i = 0; i < topology[l]+1; i++) {
        W[l][i]=&dev_W[l*topology[l]*topology[l+1]+i*topology[l+1]];//+1 Next layer
      }
    }

  float ***DW;
  DW = new float**[HIDDEN_LAYERS+1];
  for(int l = 0; l < HIDDEN_LAYERS+1; l++){
    DW[l] = new float*[topology[l]+1];//+1 for bias
    for (int i = 0; i < topology[l]+1; i++) {
      DW[l][i]=new float[topology[l+1]];//+1 Next layer
      for(int j=0;j<topology[l+1];j++){
        DW[l][i][j]=0;
      }
    }
  }
  float **l_net;//after activation
  ////  Inicializando salida de neuronas
  l_net = new float*[HIDDEN_LAYERS+1];
  for (int i = 0; i < HIDDEN_LAYERS+1; i++) {
    l_net[i]=new float[topology[i+1]];
  }

  float **l_out;//before activation
  ////  Inicializando salida de activacion
  //// Se inicializará el elemento 0 con la data de entrenamiento asi que
  //// aumenta el tamaño en 1
  l_out = new float*[HIDDEN_LAYERS+2];
  for (int i = 0; i < HIDDEN_LAYERS+2; i++) {
    l_out[i]=new float[topology[i]];
  }
  ///////////////////////////////////////////////////////
                  /*ENTRENAMIENTO*/
  //////////////////////////////////////////////////////
  float error_epoch;

  for (int ep = 0; ep < EPOCHS; ep++) {
    if (ep%100==0 && offset==0) {
      printf("Epoca %u ------",ep);
      //cout<<"Epoca "<<ep<<"------Error: "<<error_epoch<<endl;
    }
    error_epoch=0;
    //Reiniciando actualizacion de pesos
    for (int m = offset; m < TRAIN_DSIZE; m+=N) {
      for (int l = 0; l < HIDDEN_LAYERS+1; l++) {
        for (int i = 0; i < topology[l]+1; i++) {
          for (int j = 0; j < topology[l+1]; j++) {
            DW[l][i][j]=0;
          }
        }
      }
      int m_data=m*BATCHSIZE+offset;

      ///////////////////////////////////////////////////////
                      /*FORWARD_TRAINING*/
      //////////////////////////////////////////////////////

      l_out[0]=&X_TRAIN[m_data*INPUT_SIZE];
      //l_out[0]=X_TRAIN[m];
      for (int l = 0; l < HIDDEN_LAYERS+1; l++) {
        for (int j = 0; j < topology[l+1]; j++) {
          l_net[l][j]=W[l][0][j];
          /// No cuenta el bias. El ultimo elemento incrementa indice en 1.
          for (int i = 0; i < topology[l]; i++) {
            l_net[l][j]+=W[l][i+1][j]*l_out[l][i];
          }
          l_out[l+1][j]=sigmoid(l_net[l][j]);
        }
      }
      /////////// Calculo del error por iteraacion y epoca
      float error=0;
      for (int i = 0; i < OUTPUT_SIZE; i++) {
        error+=0.5*pow(Y_TRAIN[m*OUTPUT_SIZE+i]-l_out[HIDDEN_LAYERS+1][i],2);
      }

      //atomicAdd(&error_epoch,error);
      error_epoch+=error;



      ///////////////////////////////////////////////////////
                      /*BACKWARD_TRAINING*/
      //////////////////////////////////////////////////////
        /////////// Calculo para output layer
        for (int j = 0; j < topology[HIDDEN_LAYERS+1]; j++) {
          float prediccion=l_out[HIDDEN_LAYERS+1][j];
          //printf("Calculo Bias %u:\n",j);
          DW[HIDDEN_LAYERS][0][j]=prediccion-Y_TRAIN[m_data*OUTPUT_SIZE+j];
          DW[HIDDEN_LAYERS][0][j]*=dsigmoid(prediccion);
          for (int i = 0; i < topology[HIDDEN_LAYERS]; i++) {
            DW[HIDDEN_LAYERS][i+1][j]=DW[HIDDEN_LAYERS][0][j]*l_out[HIDDEN_LAYERS][i];
          }
        }

        if(HIDDEN_LAYERS>0){
          ///////////CALCULO HIDDEN LAYERS
          for (int l = HIDDEN_LAYERS-1; l >-1; l--) {
            for (int j = 0; j < topology[l+1]; j++) {
              ////// Calculando actualizaciones de bias//////////
              float sum=0;
              for (int k = 0; k < topology[l+2]; k++) {
                sum+=W[l+1][j+1][k]*DW[l+1][0][k];
              }
              //printf("SUMA: %f\n",sum);
              DW[l][0][j]=sum*dsigmoid(l_out[l+1][j]);
              //printf("Dsigmoid: %f\n",dsigmoid(this->l_out[l+1][j]));
              //printf("Bias %u: %f\n",j,this->theta[l].DW[0][j]);
              for (int i = 0; i < topology[l]; i++) {
                DW[l][i+1][j]=DW[l][0][j]*l_out[l][i];
                //printf("DW%u%u: %f\n",i-1,j,this->theta[l].DW[i][j]);
              }
            }
          }
        }
        __syncthreads();
        for (int l = 0; l < HIDDEN_LAYERS+1; l++) {
          for (int i = 0; i < topology[l]+1; i++) {
            for (int j = 0; j < topology[l+1]; j++) {
              atomicAdd(&W[l][i][j],-LEARNING_RATE*DW[l][i][j]);
            }
          }
        }
        //atomicAdd(&error_epoch,error);
    }
    if (ep%100==0 && offset==0) {
      printf("Error: %f\n",error_epoch);
      //cout<<"Epoca "<<ep<<"------Error: "<<error_epoch<<endl;
    }
  }
  }
}


int main() {

  ///////////////////////////////////////////////////////
                  /*LEYENDO DATA*/
  //////////////////////////////////////////////////////

  //float X_TRAIN[TRAIN_DSIZE][INPUT_SIZE];
  //float Y_TRAIN[TRAIN_DSIZE][OUTPUT_SIZE]={0};
  string row;
  string data_aux;
  ifstream file("weights.txt");
  printf("Abriendo...\n");
  int aux_img1=0;
  int aux_img2=0;
  for (int img = 0; img < DATA_SIZE; img++) {
    getline (file,row,'\n');
    stringstream ss(row);
    if (img%5==0) {
      for (int w = 0; w < INPUT_SIZE; w++) {
        getline (ss,data_aux,',');
        X_TEST[aux_img1*INPUT_SIZE+w]=stod(data_aux)/40000;
      }
    getline(ss,data_aux,',');
    Y_TEST[aux_img1*INPUT_SIZE+stoi(data_aux)-1]=1;
    //cout<<"testing output "<<stoi(data_aux)-1<<" : "<<Y_TEST[aux_img1][stoi(data_aux)-1]<<endl;
    aux_img1++;
    }
    else{
      for (int w = 0; w < INPUT_SIZE; w++) {
        getline (ss,data_aux,',');
        X_TRAIN[aux_img2*INPUT_SIZE+w]=stod(data_aux)/40000;
      }
      getline(ss,data_aux,',');
      Y_TRAIN[aux_img2*INPUT_SIZE+stoi(data_aux)-1]=1;
      aux_img2++;
      //cout<<"training output "<<stoi(data_aux)-1<<" : "<<Y_TRAIN[aux_img2][stoi(data_aux)-1]<<endl;
    }

  }

  cout<<"Y_train=[ ";
  for (int i = 0; i < TRAIN_DSIZE; i++) {
    for (int j = 0; j < OUTPUT_SIZE; j++) {
      cout<<Y_TRAIN[i*OUTPUT_SIZE+j]<<" ";
    }
    cout<<"fila "<<i<<endl;
  }
  cout<<"]"<<endl;
  ///////////////////////////////////////////////////////
                  /*Generando topología*/
  //////////////////////////////////////////////////////

  topology[0]=INPUT_SIZE;
  for (int i = 1; i < HIDDEN_LAYERS+1; i++) {
    topology[i]=HIDDEN_SIZE;
  }
  topology[HIDDEN_LAYERS+1]=OUTPUT_SIZE;
  printf("Topology: ");
  for (int i = 0; i < HIDDEN_LAYERS+2; i++) {
    printf("%u ",topology[i] );
  }
  printf("\n");
  printf("TOPOLOGIA GENERADA\n");

  ///////////////////////////////////////////////////////
                  /*Generando RED*/
  //////////////////////////////////////////////////////

  int size3=0;

  for(int i=0;i<HIDDEN_LAYERS+1;i++)
    size3+=(topology[i]+1)*topology[i+1];
  cout<<"size3: "<<size3<<endl;
  float* W = new float[size3];

  for (int i=0;i<size3;i++){
    W[i]=0.2*(float(rand()) / float(RAND_MAX))/2.f;;
  }

  cout<<"Red creada"<<endl;
  ///////////////////////////////////////////////////////
                  /*Comprobando RED*/
  //////////////////////////////////////////////////////


  for (int l = 0; l < HIDDEN_LAYERS+1; l++) {
    printf("THETA %u:\n",l);
    printf("Size i: %u x o: %u\n", topology[l], topology[l+1]);

    for (int row = 0; row < topology[l]; row++) {
      for (int col = 0; col < topology[l+1]; col++) {
        printf("%f ",W[l*(topology[l]+1)*topology[l+1]+row*(topology[l]+1)+col]);
      }
      printf("\n");
    }
  }
  ///////////////////////////////////////////////////////
                  /*Reservacion de espacios*/
  //////////////////////////////////////////////////////
  float *dev_W;
  cudaMalloc(&dev_W, size3*sizeof(float));
  cudaMemcpy( dev_W, W, size3*sizeof(float), cudaMemcpyHostToDevice);

  int *dev_topology;
  cudaMalloc(&dev_topology,(HIDDEN_LAYERS+2)*sizeof(int));
  cudaMemcpy( dev_topology, topology, (HIDDEN_LAYERS+2)*sizeof(int), cudaMemcpyHostToDevice);

  float *dev_X_TRAIN;
  cudaMalloc(&dev_X_TRAIN,INPUT_SIZE*TRAIN_DSIZE*sizeof(float));
  cudaMemcpy( dev_X_TRAIN, X_TRAIN, INPUT_SIZE*TRAIN_DSIZE*sizeof(float), cudaMemcpyHostToDevice);

  float *dev_Y_TRAIN;
  cudaMalloc(&dev_Y_TRAIN,INPUT_SIZE*TRAIN_DSIZE*sizeof(float));
  cudaMemcpy( dev_Y_TRAIN, Y_TRAIN, INPUT_SIZE*TRAIN_DSIZE*sizeof(float), cudaMemcpyHostToDevice);

  float *c,*dev_c;
  c = (float*)malloc(4*sizeof(int));
  cudaMalloc(&dev_c, 4*sizeof(float));
  cudaMemcpy( dev_c, c, 4*sizeof(float), cudaMemcpyHostToDevice);

  ///////////////////////////////////////////////////////
                  /*Entrenamiento*/
  //////////////////////////////////////////////////////
  int blocksize= BATCHSIZE;
  int gridsize= (N-1+blocksize)/blocksize;
  ////////////  Iteraciones  ///////////////////////////
  //float start = omp_get_wtime( );
  //void training(float *dev_W,int *topology,float *X_TRAIN,float *Y_TRAIN)
  cout<<"Llega kernel"<<endl;
  training<<<gridsize,blocksize>>>(dev_W,dev_topology,dev_X_TRAIN,dev_Y_TRAIN);
  cudaMemcpy( c, dev_c, 4*sizeof(float), cudaMemcpyDeviceToHost );
  //float end = omp_get_wtime( );
  //printf("time = %f s\n",(end-start));
  //testing();
  printf("TESTEO TERMINADO\n");
}
