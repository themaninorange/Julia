//nvcc fractalAnimPath.cu -o temp -lglut -lGL -lm -run

#include <GL/glut.h>
#include <stdlib.h>
#include <iostream>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <signal.h>

#define PI 3.14159265359

using namespace std;

float *pixelsJulia_CPU, *pixelsMandel_CPU, *pixels_CPU;

float *pixelsJulia_GPU, *pixelsMandel_GPU, *pixels_GPU;

//dim3 dimBlock;

/*float A = -0.624;
float B = 0.4351; // */

float A = 0;
float B = 0; // */
float t = 0;
float tmod = 100.0;
float titer = 0.01;
float moveiter = 1.0;
int N = 100;
int mandelID; 
int juliaID; 
char pathChoice = 'A';
unsigned int window_height = 960;
unsigned int window_width = 2*window_height;

float xMin = -2.0;
float xMax =  2.0;
float yMin = -2.0;
float yMax =  2.0;

float stepSizeX = (xMax - xMin)/((float)window_width);
float stepSizeY = (yMax - yMin)/((float)window_height);

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
	}// */
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
}// */


__global__ void cudaColorJulia(float *pixelsJulia_GPU, float X, float iY){
	
	float x = (((float)threadIdx.x)/(blockDim.x))*4-2;
	float y = (((float)blockIdx.x)/(gridDim.x))*4-2;
	float mag,maxMag, t1;
	int maxCount = 200;
	int count = 0;
	maxMag = 10;
	mag = 0.0;

	while (mag < maxMag && count < maxCount) 
	{
		t1 = x;	
		x = x*x - y*y + X;
		y = (2.0 * t1 * y) + iY;
		mag = sqrt(x*x + y*y);
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

	}// */
	

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
}// */

float timeChange(char pathChoice){
	
	switch(pathChoice){
		case 'A' :
			if(t < 0.02){
				return(titer*t + 0.000001);
			} else if(t < 0.1){
				return(1.8*titer*t + 0.02 + 0.000001);
			} else if(t < 0.6){
				return(0.2*titer*0.1 + 0.000001);
			} else if(t < 0.805){
				return(0.5*titer*(fabs(t - 0.7025 )/0.205) + 0.000001);
			} else{
				return(0.5*titer*((-t + 1)/0.25) + 0.000001);
			}
			break;
		case 'B' :
			return(1);
			break;
		case 'C' :
			return(1);
	}

}

void pathMove(char pathChoice){
	
	switch(pathChoice){
		case 'A' :
			A = 0.5*sin(t*(2*PI)) + 0.25;
			B = -0.5*cos(t*(2*PI)) + 0.5;
			break;
		case 'B' :
			A = t;
			B = t*t;
			break;
		case 'C' :
			A = 1;
			B = 2;
	}
	
}

void update(int value){
	
	t = t + timeChange(pathChoice);
	if(t >= 1){
		t -= 1;
	}
	
	pathMove(pathChoice);
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
			A = A + moveiter/window_width;	break;
		case GLUT_KEY_LEFT :
			A = A - moveiter/window_width;	break;
		case GLUT_KEY_UP :
			B = B + moveiter/window_height;	break;
		case GLUT_KEY_DOWN :
			B = B - moveiter/window_height;	break;
	}
	
}// */

void processNormalKeys(unsigned char key, int x, int y) {

	if(key == 43){		//Plus sign key, '+'
		titer = titer * 1.2;
	}else if(key == 45){	//Minus sign key, '-'
		titer = titer/1.2;
	}
}// */

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
			B = -(float)y/window_height*4.0+2.0;	break;
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

	cudaColorJulia<<<window_width/2, window_height>>>(pixelsJulia_GPU, A, B);
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

