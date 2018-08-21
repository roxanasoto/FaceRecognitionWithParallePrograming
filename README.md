# FaceRecognitionWithParallePrograming
Sistema de Reconocimiento de Rostros usando Programación paralela OPENMP

Para la ejecución de este propgrama se hace uso del sistema operativo Ubuntu.

Debe tener Opencv, make instalado

El proyecto esta divido en dos fases.

Fase 1: Extracción de características usando PCA

su implementación esta en la carpeta "Fase1_ExtracciónCaracterísticasPCA" para la ejecución de esta carpeta debe entrar a dicha carpeta y ejecutar los siguientes comandos:

$make

$./facial_features

esto le generará los pesos que deben estar ubicados en la carpeta 

FaceRecognitionWithParallePrograming/Fase1_ExtracciónCaracterísticasPCA/dataset/weights

Estos servirá de entrada para la siguiente fase

Fase 2: Reconocmiento usando la red neuronal Back Propagation

su implementacion 
