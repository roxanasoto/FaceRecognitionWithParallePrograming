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
#include <omp.h>

using namespace std;

#define TRAIN_DSIZE 150 // Numero de datos de entrenamiento
#define TEST_DSIZE 150 // Numero de datos de testeo
#define INPUT_SIZE 15 // Numero de neuronas en capa de entrada
#define HIDDEN_LAYERS 1 // Numero de capas ocultas
#define HIDDEN_SIZE 30 // Numero de neuronas en capas ocultas
#define OUTPUT_SIZE 15 //Numero de neuronas en capa de salida
#define LEARNING_RATE 0.1 //Tasa de aprendizaje
#define EPOCHS 40000 //Numero de epocas

double error; // Error por iteracion : usado para backpropagation
double error_epoch; //Error por epoca : usado para visualizacion
int topology[HIDDEN_LAYERS+2]; // Numero de neuronas por capa (entrada, ocultas y salida)
double X_TRAIN[TRAIN_DSIZE][INPUT_SIZE]; // Entrada de entrenamiento
double Y_TRAIN[TRAIN_DSIZE][OUTPUT_SIZE]={0}; //Salida de entrenamiento
int waiting=0;

// Estructuta para pesos entre cada par de capas consecutivas.La estructura bpnet contiene
// un array de estructuras l_theta.
struct l_theta
{
  int i_size; // Numero de neuronas en capa de entrada
  int o_size; // Numero de neuronas en capa de salida
  int ind; // Indice de la estructura en bpnet
  double **W; // Array de pesos de orden i_size x o_size
  double **DW; // Usado para guardar las actualizaciones de los pesos hasta el final de la iteracion
  double *l_output;// Salida de neuronas sin pasar por capa de activacion
  double *l_delta; //Error usado para actualizar DW en backpropagation

  void create(int topology[HIDDEN_LAYERS+2]){ //HIDDEN_LAYERS+ INPUT LAYER + OUTPUT LAYER
    //Inicializacion de la estructura segun la topologia definida
    this->i_size=topology[ind]+1; //Se agrega un elemento en entrada para el bias
    this->o_size=topology[ind+1];
    this->W = new double*[i_size];
    for(int i = 0; i < i_size; ++i){
      this->W[i] = new double[o_size];
    }
    this->DW = new double*[i_size];
    for(int i = 0; i < i_size; ++i){
      this->DW[i] = new double[o_size];
    }
    this->l_output = new double[o_size];
    this->l_delta = new double[o_size];

    //Inicializando THETA con valores aleatorios
    for (int i = 0; i < i_size; i++) {
      for (int j = 0; j < o_size; j++) {
        this->W[i][j]=0.2*(double(rand()) / double(RAND_MAX))/2.f;
      }
    }
  }

};

struct bpnet
{
  double **in_forward;// Salida de activacion para cada neurona desde la primera capa oculta
  struct l_theta theta[HIDDEN_LAYERS+1]; // Array de estructuras de pesos
  void create(int topology[HIDDEN_LAYERS+2]){
    // Inicializacion de las estructuras theta
    for (int i=0;i<HIDDEN_LAYERS+1;i++){
      theta[i].ind=i;
      theta[i].create(topology);
    }
    //Inicializacion in_forward
    this->in_forward = new double*[HIDDEN_LAYERS+2];
    for(int i = 0; i < HIDDEN_LAYERS+2; i++){
      this->in_forward[i] = new double[topology[i]];
    }
  };

  double sigmoid(double x) {
  	return (1.0f / (1.0f + exp(-x)));
  }

  double dsigmoid(double x) {
  	return (x*(1-x));
  }

  void forward(int m,int myrank){
    //printf("FORWARD\n");
    this->in_forward[0]=X_TRAIN[m];
    for (int l = 0; l < HIDDEN_LAYERS+1; l++) {

      for (int j = 0; j < theta[l].o_size; j++) {
        this->theta[l].l_output[j]=this->theta[l].W[0][j];
        for (int i = 1; i < theta[l].i_size; i++) {
          this->theta[l].l_output[j]+=this->theta[l].W[i][j]*this->in_forward[l][i-1];
        }
        this->in_forward[l+1][j]=sigmoid(this->theta[l].l_output[j]);

      }
    }
    ///////////CALCULO ERROR/////////////////////
    double error=0;
    for (int i = 0; i < OUTPUT_SIZE; i++) {
      error+=0.5*pow(Y_TRAIN[m][i]-this->in_forward[HIDDEN_LAYERS+1][i],2);
    }
    error_epoch+=error;
  }

  void forward_test(int m){
    this->in_forward[0]=X_TRAIN[m];
    for (int l = 0; l < HIDDEN_LAYERS+1; l++) {
      for (int j = 0; j < theta[l].o_size; j++) {
        this->theta[l].l_output[j]=this->theta[l].W[0][j];
        for (int i = 1; i < theta[l].i_size; i++) {
          this->theta[l].l_output[j]+=this->theta[l].W[i][j]*this->in_forward[l][i-1];//////
        }
        this->in_forward[l+1][j]=sigmoid(this->theta[l].l_output[j]);

      }
    }
    ///////////CALCULO ERROR/////////////////////
    double error=0;
    for (int i = 0; i < OUTPUT_SIZE; i++) {
      error+=0.5*pow(Y_TRAIN[m][i]-this->in_forward[HIDDEN_LAYERS+1][i],2);
      //printf("Y: %lf\n a: %lf\n",Y_TRAIN[m][i],this->in_forward[HIDDEN_LAYERS+1][i]);
      //printf("----ERROR: %lf\n",error);
    }
    error_epoch+=error;
  }
  void backward_training(int m,int myrank){
    //printf("BACKWARD\n");
    //printf("Error: %.12lf\n",error);
    ///////////CALCULO VARIACION ULTIMA CAPA/////////////////////
    for (int j = 0; j < this->theta[HIDDEN_LAYERS].o_size; j++) {
      double prediccion=this->in_forward[HIDDEN_LAYERS+1][j];
      //printf("Calculo Bias %u:\n",j);
      this->theta[HIDDEN_LAYERS].DW[0][j]=prediccion-Y_TRAIN[m][j];
      //printf("Actualizacion:%lf\n",this->theta[HIDDEN_LAYERS].DW[0][j]);
      this->theta[HIDDEN_LAYERS].DW[0][j]*=dsigmoid(prediccion);
      //printf("Actualizacion:%lf\n",dsigmoid(prediccion));
      //printf("Bias resultado:%lf\n",this->theta[HIDDEN_LAYERS].DW[0][j]);
      for (int i = 1; i < this->theta[HIDDEN_LAYERS].i_size; i++) {
        this->theta[HIDDEN_LAYERS].DW[i][j]=this->theta[HIDDEN_LAYERS].DW[0][j]*this->in_forward[HIDDEN_LAYERS][i-1];
        //printf("Actualizacion:%lf\n",this->in_forward[HIDDEN_LAYERS][i-1]);
        //printf("\nRESULTADO_parcial:%lf\n",this->theta[HIDDEN_LAYERS].DW[i][j]);
      }
    }

    if(HIDDEN_LAYERS>0){
      ///////////CALCULO HIDDEN LAYERS/////////////////////
      //printf("CALCULO HIDDEN_LAYERS:\n");
      for (int l = HIDDEN_LAYERS-1; l >-1; l--) {
        for (int j = 0; j < this->theta[l].o_size; j++) {
          ////// Calculando actualizaciones de bias//////////
          double sum=0;
          for (int k = 0; k < this->theta[l+1].o_size; k++) {
            sum+=this->theta[l+1].W[j+1][k]*this->theta[l+1].DW[0][k];
          }
          //printf("SUMA: %lf\n",sum);
          this->theta[l].DW[0][j]=sum*dsigmoid(this->in_forward[l+1][j]);
          //printf("Dsigmoid: %lf\n",dsigmoid(this->in_forward[l+1][j]));
          //printf("Bias %u: %lf\n",j,this->theta[l].DW[0][j]);
          for (int i = 1; i < this->theta[l].i_size; i++) {
            this->theta[l].DW[i][j]=this->theta[l].DW[0][j]*this->in_forward[l][i-1];
            //printf("DW%u%u: %lf\n",i-1,j,this->theta[l].DW[i][j]);
          }
        }
      }
//#     pragma omp barrier
      ///////////ACTUALIZACION/////////////////////
      //printf("ACTUALIZACION:\n");
      for (int l = 0; l < HIDDEN_LAYERS+1; l++) {
        //printf("CAPA %u\n",l);
        for (int i = 0; i < this->theta[l].i_size; i++) {
          for (int j = 0; j < this->theta[l].o_size; j++) {
            this->theta[l].W[i][j]-=LEARNING_RATE*this->theta[l].DW[i][j];
            //printf("\nPeso %u%u:%lf\n",i,j,this->theta[l].W[i][j]);
          }
        }
      }
    }
  }
  void testing(){

    for (int m = 0; m < TEST_DSIZE; m++) {
      double pred_test_v=-1;
      int pred_test_i=-1;
      this->forward_test(m);
      for (int j = 0; j < OUTPUT_SIZE; j++) {
        if (this->in_forward[HIDDEN_LAYERS+1][j]>pred_test_v) {
          pred_test_v=this->in_forward[HIDDEN_LAYERS+1][j];
          pred_test_i=j;
          /* code */
        }
      }
      printf("Prediccion %u: %u with value %lf\n",m,pred_test_i,pred_test_v);
    }

  }

};

int main() {
  int numthreads=4;
  ///////////////////////////////////////////////////////
                  /*LEYENDO DATA*/
  //////////////////////////////////////////////////////

  //double X_TRAIN[TRAIN_DSIZE][INPUT_SIZE];
  //double Y_TRAIN[TRAIN_DSIZE][OUTPUT_SIZE]={0};
  string row;
  string data_aux;
  ifstream file("weights.txt");
  printf("Abriendo...\n");
  for (int img = 0; img < TRAIN_DSIZE; img++) {
    getline (file,row,'\n');
    stringstream ss(row);
    for (int w = 0; w < INPUT_SIZE; w++) {
      getline (ss,data_aux,',');
      X_TRAIN[img][w]=stod(data_aux)/40000;
    }
    getline(ss,data_aux,',');
    Y_TRAIN[img][stoi(data_aux)-1]=1;
  }

  ///////////////////////////////////////////////////////
                  /*Dedo topología*/
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
  struct bpnet net;
  net.create(topology);

  for (int i = 0; i < HIDDEN_LAYERS+1; i++) {
    printf("THETA %u:\n",i);
    printf("Size i: %u x o: %u\n", net.theta[i].i_size, net.theta[i].o_size);

    for (int row = 0; row < net.theta[i].i_size; row++) {
      for (int col = 0; col < net.theta[i].o_size; col++) {
        printf("%lf ",net.theta[i].W[row][col]);
      }
      printf("\n");
    }
  }
  printf("RED GENERADA\n");
  ///////////////////////////////////////////////////////
                  /*Entrenamiento*/
  //////////////////////////////////////////////////////

  ////////////  Iteraciones  ///////////////////////////
  double start = omp_get_wtime( );
  for (int ep = 0; ep < EPOCHS; ep++) {
    error_epoch=0;
    //Entrenamiento por dato
    for (int m = 0; m < TRAIN_DSIZE/numthreads; m++) {

      #   pragma omp parallel num_threads(numthreads)
      {
      int myrank=omp_get_thread_num();
      int m_k=myrank+m*numthreads;
      if (m_k<TRAIN_DSIZE) {
        net.forward(m_k,myrank);
        net.backward_training(m_k,myrank);
      }
    }

    }
    if (ep%100==0) {
      cout<<"Epoca "<<ep<<"------Error: "<<error_epoch<<endl;
    }
  }
  double end = omp_get_wtime( );
  printf("time = %lf s\n",(end-start));
  net.testing();

  printf("TESTEO TERMINADO\n");
}
