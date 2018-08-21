/*
 * Author: Roxana Soto
 *
 * A program to detect facial feature points using
 * Haarcascade classifiers for face, eyes, nose and mouth
 *
 */

#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string> 


using namespace std;
using namespace cv;

// Functions for facial feature detection
static void detectFacialFeaures(Mat&, const vector<Rect_<int> >, string, string, string);

string input_image_path;


//Normalizes a given image into a value range between 0 and 255.
Mat norm_0_255(const Mat& src)
{
// Create and return normalized image:
    Mat dst;
    switch(src.channels()) {
        case 1:
            cv::normalize(src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
            break;
        case 3:
            cv::normalize(src, dst, 0, 255, NORM_MINMAX, CV_8UC3);
            break;
        default:
            src.copyTo(dst);
            break;
    }
    return dst;
}

//Converts the images given in src into a row matrix.
Mat asRowMatrix(const vector<Mat>& src, int rtype, double alpha = 1, double beta = 0)
{
    
//Number of samples:
    size_t n = src.size();
    
//Return empty matrix if no matrices given:
    if(n == 0)
        return Mat();
    
//dimensionality of (reshaped) samples
    size_t d = src[0].total();
// Create resulting data matrix:
    Mat data (n, d, rtype);
    
// Now copy data:
    for(int i = 0; i < n; i++) {
        if(src[i].empty()) {
            string error_message = format("Image number %d was empty, please check your input data.", i);
            CV_Error(CV_StsBadArg, error_message);
        }
// Make sure data can be reshaped, throw a meaningful exception if not!
        if(src[i].total() != d) {
            string error_message = format("Wrong number of elements in matrix #%d! Expected %d was %d.", i, d, src[i].total());
            CV_Error(CV_StsBadArg, error_message);
        }
        Mat xi = data.row(i);
// Make reshape happy by cloning for non-continuous matrices:
        if(src[i].isContinuous()) {
            src[i].reshape(1, 1).convertTo(xi, rtype, alpha, beta);
        } else {
            src[i].clone().reshape(1, 1).convertTo(xi, rtype, alpha, beta);
        }
    }
    return data;
}

int main(int argc, char** argv)
{
    
    vector<Mat> db;
    
    string prefix = "face_database/";
    db.push_back(imread(prefix + "s1/1.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s1/2.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s1/3.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s1/4.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s1/5.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s1/6.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s1/7.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s1/8.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s1/9.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s1/10.pgm", IMREAD_GRAYSCALE));
        
    db.push_back(imread(prefix + "s2/1.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s2/2.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s2/3.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s2/4.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s2/5.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s2/6.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s2/7.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s2/8.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s2/9.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s2/10.pgm", IMREAD_GRAYSCALE));
    
    
    db.push_back(imread(prefix + "s3/1.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s3/2.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s3/3.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s3/4.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s3/5.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s3/6.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s3/7.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s3/8.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s3/9.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s3/10.pgm", IMREAD_GRAYSCALE));
    
    
    db.push_back(imread(prefix + "s4/1.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s4/2.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s4/3.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s4/4.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s4/5.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s4/6.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s4/7.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s4/8.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s4/9.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s4/10.pgm", IMREAD_GRAYSCALE));
    
    
    db.push_back(imread(prefix + "s5/1.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s5/2.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s5/3.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s5/4.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s5/5.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s5/6.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s5/7.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s5/8.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s5/9.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s5/10.pgm", IMREAD_GRAYSCALE));
    
    
    db.push_back(imread(prefix + "s6/1.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s6/2.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s6/3.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s6/4.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s6/5.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s6/6.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s6/7.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s6/8.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s6/9.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s6/10.pgm", IMREAD_GRAYSCALE));
    

    db.push_back(imread(prefix + "s7/1.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s7/2.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s7/3.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s7/4.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s7/5.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s7/6.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s7/7.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s7/8.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s7/9.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s7/10.pgm", IMREAD_GRAYSCALE));
    
    
    db.push_back(imread(prefix + "s8/1.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s8/2.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s8/3.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s8/4.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s8/5.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s8/6.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s8/7.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s8/8.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s8/9.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s8/10.pgm", IMREAD_GRAYSCALE));
    
    
    db.push_back(imread(prefix + "s9/1.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s9/2.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s9/3.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s9/4.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s9/5.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s9/6.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s9/7.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s9/8.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s9/9.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s9/10.pgm", IMREAD_GRAYSCALE));
    
    
    db.push_back(imread(prefix + "s10/1.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s10/2.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s10/3.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s10/4.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s10/5.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s10/6.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s10/7.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s10/8.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s10/9.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s10/10.pgm", IMREAD_GRAYSCALE));
    
    
    db.push_back(imread(prefix + "s11/1.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s11/2.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s11/3.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s11/4.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s11/5.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s11/6.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s11/7.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s11/8.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s11/9.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s11/10.pgm", IMREAD_GRAYSCALE));
    
    
    db.push_back(imread(prefix + "s12/1.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s12/2.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s12/3.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s12/4.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s12/5.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s12/6.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s12/7.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s12/8.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s12/9.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s12/10.pgm", IMREAD_GRAYSCALE));
    
    
    db.push_back(imread(prefix + "s13/1.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s13/2.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s13/3.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s13/4.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s13/5.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s13/6.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s13/7.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s13/8.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s13/9.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s13/10.pgm", IMREAD_GRAYSCALE));
    
    
    db.push_back(imread(prefix + "s14/1.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s14/2.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s14/3.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s14/4.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s14/5.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s14/6.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s14/7.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s14/8.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s14/9.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s14/10.pgm", IMREAD_GRAYSCALE));
    
    
    db.push_back(imread(prefix + "s15/1.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s15/2.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s15/3.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s15/4.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s15/5.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s15/6.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s15/7.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s15/8.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s15/9.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s15/10.pgm", IMREAD_GRAYSCALE));
    

    // Build a matrix with the observations in row:
    Mat data = asRowMatrix(db, CV_32FC1);
    
// number of components to keep for the PCA:
    int num_components = 15;
    
// Performing a PCA:
    PCA pca(data, Mat(), CV_PCA_DATA_AS_ROW, num_components);
    
//copying the PCA results:
    Mat mean = pca.mean.clone();
    Mat eigenvalues = pca.eigenvalues.clone();
    Mat eigenvectors = pca.eigenvectors.clone();
    
// The mean face:
    imshow("avg", norm_0_255(mean.reshape(1, db[1].rows)));
    
// The first 'm' eigenfaces:
    imshow("pc1", norm_0_255(pca.eigenvectors.row(0)).reshape(1, db[0].rows));
    imshow("pc2", norm_0_255(pca.eigenvectors.row(1)).reshape(1, db[1].rows));
    imshow("pc3", norm_0_255(pca.eigenvectors.row(2)).reshape(1, db[2].rows));
    imshow("pc4", norm_0_255(pca.eigenvectors.row(3)).reshape(1, db[3].rows));
    imshow("pc5", norm_0_255(pca.eigenvectors.row(4)).reshape(1, db[4].rows));
    imshow("pc6", norm_0_255(pca.eigenvectors.row(5)).reshape(1, db[5].rows));
    imshow("pc7", norm_0_255(pca.eigenvectors.row(6)).reshape(1, db[6].rows));
    imshow("pc8", norm_0_255(pca.eigenvectors.row(7)).reshape(1, db[7].rows));
    imshow("pc9", norm_0_255(pca.eigenvectors.row(8)).reshape(1, db[8].rows));
    /*imshow("pc10", norm_0_255(pca.eigenvectors.row(9)).reshape(1, db[9].rows));
    imshow("pc11", norm_0_255(pca.eigenvectors.row(10)).reshape(1, db[10].rows));
    imshow("pc12", norm_0_255(pca.eigenvectors.row(11)).reshape(1, db[11].rows));
    imshow("pc13", norm_0_255(pca.eigenvectors.row(12)).reshape(1, db[12].rows));
    imshow("pc14", norm_0_255(pca.eigenvectors.row(13)).reshape(1, db[13].rows));
    imshow("pc15", norm_0_255(pca.eigenvectors.row(14)).reshape(1, db[14].rows));
    */
    cout<<"creating files\n";

    for(int i=0;i<15;i++)//15 is the number of test
    {
        ofstream myfile;
        string str = "dataset/eigen/eigen";
        string extension = ".txt";
        string Result;
        stringstream convert;
        convert << i+1;
        Result = convert.str();
        str.append(Result);
        str.append(extension);
        cout<<str<<endl;
        myfile.open(str);
        Mat mat = eigenvectors.row(i);//.t();

        cout<<"Rows = "<<mat.rows <<"columns = "<<mat.cols<<endl;
        cv::FileStorage storage(str,cv::FileStorage::WRITE);
        //myfile<<mat;
        storage<<"img"<<mat;
        storage.release();

        myfile.close();

    }
    
    cout<<"storing process complete\n";
    

    double weight[15][150]; //75=15 test * 5 each image
    
    
//calculating the values with different eigen faces;
    for(int j=0;j<15;j++)
    {
        //double values[10304];
        double sum =0;
        size_t size;
    //write the code here to open eigen%d.txt......
        string str = "dataset/eigen/eigen";
        string extension = ".txt";
        string Result;
        stringstream convert;
        convert << j+1;
        Result = convert.str();
        str.append(Result);
        str.append(extension);
        cout<<str<<endl;
            
        ifstream file(str);
        cv::FileStorage storage(str,cv::FileStorage::READ);

        Mat values;
        storage["img"]>>values;
        storage.release();
        cout<<values.rows<<", "<<values.cols<<endl;

        /*
        while ( file.good() )
        {
            //cout<<"dentro del while"<<endl;
            getline ( file, str, ',' );
            values[i] = stod(str,&size);
            //cout <<values[i]<<endl;
            i++;
        }
        */
        
    //values for jth eigen face is avialble to me...calculated the weight of each image corresponding to this matrix
           for(int k=0;k<150;k++)
            {
                sum=0;
                int i = 0;
                for(int a=0;a<112;a++)//heigth
                {
                    for(int b=0;b<92;b++)//weight
                    {
                        double num1 = db[k].at<uchar>(a,b);
                        //cout<<"db(0)["<<a<<"]["<<b<<"] = "<<num1<<" ";
                        double num2=values.at<float>(0,i);
                      
                        double val = num1*num2;
                        //cout<<i<<", "<<a<<", "<<b<<endl;
                        sum = sum+ val;
                        i++;
                    }
                }
                cout<<endl<<"weight for "<<k+1<<" image corresponding to "<<j+1<<"th eigenface is "<<sum<<endl;
                weight[j][k] = sum;
            }

    }

   ofstream myfile;
   string str = "dataset/weights/weights.txt";
   myfile.open(str);
    //now output the weight matrix to a file
    for(int a=0;a<150;a++){
        //if((a+1)%5!=3){
       		for(int b=0;b<15;b++)
           		myfile<<weight[b][a]<<",";
        	myfile<< a / 10 + 1<<endl;
        //}
    }
    
    
    // Show the images:
    waitKey(0);
    
    return 0;
}