//I used to have make information up here.

#include <GL/glut.h>
#include <stdlib.h>
#include <iostream>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <signal.h>

using namespace std;

float *pixelsJulia_CPU, *pixelsMandel_CPU, *pixels_CPU;

float *pixelsJulia_GPU, *pixelsMandel_GPU, *pixels_GPU;

//dim3 dimBlock;


float A = 0;    //1 coordinate of seed
float B = 0.75; //i coordinate of seed
float C = 0;    //j coordinate of seed
float D = 0;    //k coordinate of seed

float vA = 0;    //1 coordinate of View
float vB = 0;    //i coordinate of View
float vC = 0;    //j coordinate of View
float vD = 0;    //k coordinate of View
float vE = 0;    //xi1 hopf coordinate of View
float vF = 0;    //xi2 hopf coordinate of View
float vG = 0;    //eta hopf coordinate of View

float t = 0;
float tmod = 100.0;
float titer = 1.0;
float moveiter = 0.001;
int N = 100;
int mandelID; 
int juliaID; 
unsigned int window_height = 960;
unsigned int window_width = 2*window_height;

/*
float xMin = -2.0;
float xMax =  2.0;
float yMin = -2.0;
float yMax =  2.0;

float stepSizeX = (xMax - xMin)/((float)window_width);
float stepSizeY = (yMax - yMin)/((float)window_height);
*/

void AllocateMemory(){
	cudaMalloc((void**)&pixelsJulia_GPU, window_width/2*window_height*3*sizeof(float));
	pixelsJulia_CPU = (float *)malloc(window_width/2*window_height*3*sizeof(float));
	cudaMalloc((void**)&pixelsMandel_GPU, window_width/2*window_height*3*sizeof(float));
	pixelsMandel_CPU = (float *)malloc(window_width/2*window_height*3*sizeof(float));
	cudaMalloc((void**)&pixels_GPU, window_width*window_height*3*sizeof(float));
	pixels_CPU = (float *)malloc(window_width*window_height*3*sizeof(float));

} // */	//Saves the appropriate memory chunks for later use.
	//References the globally defined variables.

float color (float x, float y)	//hopefully centered on (0,0)? 
{
	float mag,maxMag,t1;
	float maxCount = 200;
	float count = 0;
	maxMag = 10;
	mag = 0.0;

	while (mag < maxMag && count < maxCount) 
	{
		t1 = x;	
		x = x*x - y*y + A;
		y = (2.0 * t1 * y) + B;
		mag = sqrt(x*x + y*y);
		count++;
	}
	if(count < maxCount) 
	{
		return(1.0);
	}
	else
	{
		return(0.0);
	}
}

__global__ void cudaWeave(float *pixelsMandel_GPU, float *pixelsJulia_GPU, float *pixels_GPU){
	
	//red
	pixels_GPU[(2*blockIdx.x*blockDim.x + threadIdx.x)*3] = 
		pixelsJulia_GPU[(blockIdx.x*blockDim.x + threadIdx.x)*3];
		//First 600 on each row should be from the Julia set.
	
	pixels_GPU[((2*blockIdx.x+1)*blockDim.x + threadIdx.x)*3] = 
		pixelsMandel_GPU[(blockIdx.x*blockDim.x + threadIdx.x)*3];	
		//601-1200 on each row should be from the Mandelbrot set.

	//green
	pixels_GPU[(2*blockIdx.x*blockDim.x + threadIdx.x)*3+1] = 
		pixelsJulia_GPU[(blockIdx.x*blockDim.x + threadIdx.x)*3+1];
		//""
	pixels_GPU[((2*blockIdx.x+1)*blockDim.x + threadIdx.x)*3+1] = 
		pixelsMandel_GPU[(blockIdx.x*blockDim.x + threadIdx.x)*3+1];	
		//""	
	//blue
	pixels_GPU[(2*blockIdx.x*blockDim.x + threadIdx.x)*3+2] = 
		pixelsJulia_GPU[(blockIdx.x*blockDim.x + threadIdx.x)*3+2];
		//""
	pixels_GPU[((2*blockIdx.x+1)*blockDim.x + threadIdx.x)*3+2] = 
		pixelsMandel_GPU[(blockIdx.x*blockDim.x + threadIdx.x)*3+2];	
		//""
}


__global__ void cudaColorJulia(float *pixelsJulia_GPU, float A, float B, float C, float D){
	
	float a = (((float)threadIdx.x)/(blockDim.x))*4-2;
	float b = (((float)blockIdx.x)/(gridDim.x))*4-2;
	float c = 0;
	float d = 0;
	
	float mag,maxMag, ta;
	int maxCount = 200;
	int count = 0;
	maxMag = 10;
	mag = 0.0;

	while (mag < maxMag && count < maxCount) 
	{
		ta = a;
			
		a = a*a - b*b - c*c - d*d + A;
		b = 2*ta*b+B;
		c = 2*ta*c+C;
		d = 2*ta*d+D;
		mag = sqrt(a*a + b*b + c*c + d*d);
		count++;
	}
	if(count < maxCount) 
	{
		pixelsJulia_GPU[(blockIdx.x*blockDim.x + threadIdx.x)*3] = 
			0.5*log((double)count)/log((double)maxCount); 
		pixelsJulia_GPU[(blockIdx.x*blockDim.x + threadIdx.x)*3 + 1] =
			1.0*log((double)count)/log((double)maxCount); 
		pixelsJulia_GPU[(blockIdx.x*blockDim.x + threadIdx.x)*3 + 2] = 
			0.4;
	}
	else
	{
		pixelsJulia_GPU[(blockIdx.x*blockDim.x + threadIdx.x)*3] = 0.0;
		pixelsJulia_GPU[(blockIdx.x*blockDim.x + threadIdx.x)*3 + 1] = 0.0;
		pixelsJulia_GPU[(blockIdx.x*blockDim.x + threadIdx.x)*3 + 2] = 0.0;

	}
	

}

__global__ void cudaColorMandelbrot(float *pixelsMandel_GPU, float xseed, float yseed){
	
	float x = 0;
	float y = 0;
	float X = (((float)threadIdx.x)/(blockDim.x))*4-2.5;
	float iY = (((float)blockIdx.x)/(gridDim.x))*4-2;
	float mag,maxMag, t1;
	int maxCount = 200;
	int count = 0;
	maxMag = 10;
	mag = 0.0;
	
	if ((abs(xseed - (float)threadIdx.x/blockDim.x*4+2.5) <= 2.0/blockDim.x) && (abs(yseed - (float)blockIdx.x/gridDim.x*4+2)) <=2.0/gridDim.x){
	//If this pixel corresponds to the seed for the Julia set being generated,
		pixelsMandel_GPU[(blockIdx.x*blockDim.x + threadIdx.x)*3] = 1.0;
		pixelsMandel_GPU[(blockIdx.x*blockDim.x + threadIdx.x)*3 + 1] = 0.0;
		pixelsMandel_GPU[(blockIdx.x*blockDim.x + threadIdx.x)*3 + 2] = 0.0;
		//... make this pixel red.
	}
	else{
	//Otherwise, find the color the way we normally would with the Mandelbrot set.
	
		while (mag < maxMag && count < maxCount){
		//As long as the complex number doesn't get farther than a certain distance,
		// and as long as we haven't iterated this enough times,
			t1 = x;	
			x = x*x - y*y + X;
			y = (2.0 * t1 * y) + iY;
			mag = sqrt(x*x + y*y);
			count++;
			//... find the next point in the sequence and update to it.
		}
		if(count < maxCount){
		//If we broke the above loop before iterating as many times as we want,
		// then the sequence diverges,
		
			pixelsMandel_GPU[(blockIdx.x*blockDim.x + threadIdx.x)*3] = 
				0.5*log((double)count)/log((double)maxCount); 
			pixelsMandel_GPU[(blockIdx.x*blockDim.x + threadIdx.x)*3 + 1] =
				1.0*log((double)count)/log((double)maxCount); 
			pixelsMandel_GPU[(blockIdx.x*blockDim.x + threadIdx.x)*3 + 2] = 
				0.4;
			//... and we color it prettily according to how quickly it diverged.
		}
		else
		//Otherwise, the point is in the mandelbrot set (or close enough to it),
		{
			pixelsMandel_GPU[(blockIdx.x*blockDim.x + threadIdx.x)*3] = 0.0;
			pixelsMandel_GPU[(blockIdx.x*blockDim.x + threadIdx.x)*3 + 1] = 0.0;
			pixelsMandel_GPU[(blockIdx.x*blockDim.x + threadIdx.x)*3 + 2] = 0.0;
			//... and we color it black.
		}
	}
}
void update(int value){
	
	//t = t + titer;
	/*A = -pow(sin(t/tmod),2);
	B = sin(2*t/tmod)/2;// */
	/*A = -pow(sin((2*t/5)/50.0),2);
	B = sin((2*t/3)/50.0)/2;// */
	/*A = -pow(sin((2.0*t/5)/tmod),2);
	B = sin((sqrt(5)+1)/2*(t)/tmod)/2;// */
	glutPostRedisplay();
	glutTimerFunc(16,update, 0);
}

/*static void signalHandler(int signum) {
	int command;
	bool exitMenu = 0;
	//cout << "I handled it :)" << endl;
	
	while (exitMenu == 0) {
	    	cout << "Enter 0 to exit the program." << endl;
		cout << "Enter 1 to continue." << endl;
		cin >> command;
		
		if(command == 0) {
			exitMenu = 1;
		} else if (command == 1){
			exitMenu = 1;
			cout << "resuming..." << endl;
		} else {
			cout << "Invalid Command!" << endl;
		}
		cout << endl;
	}
}// */

/*void processSpecialKeys(int key, int x, int y) {

	switch(key) {
		case GLUT_KEY_RIGHT :
				t = t + titer*100;	break;
		case GLUT_KEY_LEFT :
				t = t - titer*100;	break;
		case GLUT_KEY_UP :
				titer = titer*1.1;	break;
		case GLUT_KEY_DOWN :
				titer = titer/1.1;	break;
	}
}// */

void processSpecialKeys(int key, int x, int y) {

	switch(key) {
		case GLUT_KEY_RIGHT :
			A = A + moveiter;	break;
		case GLUT_KEY_LEFT :
			A = A - moveiter;	break;
		case GLUT_KEY_UP :
			B = B + moveiter;	break;
		case GLUT_KEY_DOWN :
			B = B - moveiter;	break;
	}
	
}

void processNormalKeys(unsigned char key, int x, int y) {
    
    switch(key) {
	    case 43:	//Plus sign key, '+'
		    moveiter = moveiter * 1.2;  break;
	    case 45:	//Minus sign key, '-'
		    moveiter = moveiter/1.2;    break;
	    case 81:	//Q = 17 + 64
	    case 113:	//q = 17 + 96
		    A = A + moveiter;   break;
	    case 65:	//A = 1  + 64
	    case 97:	//a = 1  + 96
		    A = A - moveiter;   break;
	    case 87:	//W = 23 + 64
	    case 119:	//w = 23 + 96
		    B = B + moveiter;   break;
	    case 83:	//S = 19 + 64
	    case 115:	//s = 19 + 96
		    B = B - moveiter;   break;
	    case 69:	//E = 5  + 64
	    case 101:	//e = 5  + 96
		    C = C + moveiter;   break;
	    case 68:	//D = 4  + 64
	    case 100:	//d = 4  + 96
		    C = C - moveiter;   break;
	    case 82:	//R = 18 + 64
	    case 114:	//r = 18 + 96
		    D = D + moveiter;   break;
	    case 70:	//F = 6  + 64
	    case 102:	//f = 6  + 96
		    D = D - moveiter;   break;
	    case 84:	//T = 20 + 64
	    case 116:	//t = 20 + 96
		    D = D - moveiter;   break;
	    case 71:	//G = 7  + 64
	    case 103:	//g = 7  + 96
		    D = D - moveiter;   break;
	    case 89:	//Y = 25 + 64
	    case 121:	//y = 25 + 96
		    D = D - moveiter;   break;
	    case 72:	//H = 8  + 64
	    case 104:	//h = 8  + 96
		    D = D - moveiter;   break;
	    case 85:	//U = 21 + 64
	    case 117:	//u = 21 + 96
		    D = D - moveiter;   break;
	    case 74:	//J = 10 + 64
	    case 106:	//j = 10 + 96
		    D = D - moveiter;   break;
	    case 73:	//I = 9  + 64
	    case 105:	//i = 9  + 96
		    D = D - moveiter;   break;
	    case 75:	//K = 11 + 64
	    case 107:	//k = 11 + 96
		    D = D - moveiter;   break;
	    case 79:	//O = 15 + 64
	    case 111:	//o = 15 + 96
		    D = D - moveiter;   break;
	    case 76:	//L = 12 + 64
	    case 108:	//l = 12 + 96
		    D = D - moveiter;   break;
	    case 80:	//P = 16 + 64
	    case 112:	//p = 16 + 96
		    D = D - moveiter;   break;
	    case 59:	//; = 59
	    case 58:	//: = 58
		    D = D - moveiter;   break;
	    case 91:	//[ = 91
	    case 123:	//{ = 123
		    D = D - moveiter;   break;
	    case 44:	//' = 44
	    case 34:	//" = 34
		    D = D - moveiter;   break;
	}
}

void mouseClicks(int button, int state, int x, int y) {

	/*switch(button) {
		case GLUT_LEFT_BUTTON :
			A = ((float)x-window_width/2)/window_width*2.0-2.5;
			B = -(float)y/window_height*2.0-2.0;	break;
		case GLUT_RIGHT_BUTTON :
			break;
	}*/
	
	switch(button) {
		case GLUT_LEFT_BUTTON :
			A = ((float)x)/window_width*8.0-6.5;
			B = (-(float)y)/window_height*4.0+2.0;	break;
		case GLUT_RIGHT_BUTTON :
			break;
	}

	
}// */

/*void displayJulia(void) 
{ 
	glutSetWindow(juliaID);
	cudaColorJulia<<<window_width, window_height>>>(pixelsJulia_GPU, A, B);
	cudaMemcpy(pixelsJulia_CPU, pixelsJulia_GPU,
		window_width*window_height*3*sizeof(float),
		cudaMemcpyDeviceToHost);
	glDrawPixels(window_width, window_height, GL_RGB, GL_FLOAT, pixelsJulia_CPU); 
	glFlush(); 
	
}

void displayMandelbrot(void) 
{
	glutSetWindow(mandelID);
	cudaColorMandelbrot<<<window_width, window_height>>>(pixelsMandel_GPU, A, B);
	cudaMemcpy(pixelsMandel_CPU, pixelsMandel_GPU,
		window_width*window_height*3*sizeof(float),
		cudaMemcpyDeviceToHost);
	glDrawPixels(window_width, window_height, GL_RGB, GL_FLOAT, pixelsMandel_CPU); 
	glFlush(); 
}// */


void weavePixels(){
	
	cudaMemcpy(pixelsJulia_GPU, pixelsJulia_CPU,
		window_width/2*window_height*3*sizeof(float),
		cudaMemcpyHostToDevice);
	cudaMemcpy(pixelsMandel_GPU, pixelsMandel_CPU,
		window_width/2*window_height*3*sizeof(float),
		cudaMemcpyHostToDevice);
	
	cudaWeave<<<window_width/2,window_height>>>(pixelsMandel_GPU, pixelsJulia_GPU, pixels_GPU);

	cudaMemcpy(pixels_CPU, pixels_GPU,
		window_width*window_height*3*sizeof(float),
		cudaMemcpyDeviceToHost);

}// */

void display(void){

	cudaColorJulia<<<window_width/2, window_height>>>(pixelsJulia_GPU, A, B, C, D);
	cudaMemcpy(pixelsJulia_CPU, pixelsJulia_GPU,
		window_width/2*window_height*3*sizeof(float),
		cudaMemcpyDeviceToHost);
	cudaColorMandelbrot<<<window_width/2, window_height>>>(pixelsMandel_GPU, A, B);
	cudaMemcpy(pixelsMandel_CPU, pixelsMandel_GPU,
		window_width/2*window_height*3*sizeof(float),
		cudaMemcpyDeviceToHost);

	weavePixels();

/*	//glRasterPos2i(0,0);
	glDrawPixels(window_width/2, window_height, GL_RGB, GL_FLOAT, pixelsJulia_CPU);
	//glRasterPos2i(window_width/2,0);
	glDrawPixels(window_width/2, window_height, GL_RGB, GL_FLOAT, pixelsMandel_CPU); // */

	glDrawPixels(window_width, window_height, GL_RGB, GL_FLOAT, pixels_CPU);
    /*
    glFontBegin(&font);
    glScalef(8.0, 8.0, 8.0);
    glTranslatef(30, 30, 0);
    glFontTextOut("Test", 5, 5, 0);
    glFontEnd();
    */
	glFlush(); 


}

void CleanUp(float *A_CPU, float *B_CPU, float *C_CPU, 
	     float *A_GPU, float *B_GPU, float *C_GPU){
	free(A_CPU);
	free(B_CPU);
	free(C_CPU);

	cudaFree(A_GPU);
	cudaFree(B_GPU);
	cudaFree(C_GPU);
} // */	


int main(int argc, char *argv[])
{ 
	if(argc == 2){
		char *ptr;
		N = strtol(argv[1], &ptr, 10);
	}
	else if(argc > 2){
		printf("One or zero arguments expected.");
		return(1);
	}
	
	AllocateMemory();
	
   	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);
   	glutInitWindowSize(window_width, window_height);
   	
   	/*glutCreateWindow("Subwindow");
   	glutDisplayFunc(displayMandelbrot);
   	mandelID = glutGetWindow();
   	glutMouseFunc(mouseClicks);
	glutSpecialFunc(processSpecialKeys);
	glutKeyboardFunc(processNormalKeys);// */
	
   	glutCreateWindow("Fractals man, fractals.");
   	glutDisplayFunc(display);
   	//juliaID = glutGetWindow();
   	glutMouseFunc(mouseClicks);
	//glutSpecialFunc(processSpecialKeys);
	glutKeyboardFunc(processNormalKeys);
	glutSpecialFunc(processSpecialKeys);
	glutTimerFunc(16, update, 0);
	glutMainLoop();
}

