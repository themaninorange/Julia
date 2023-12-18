//I used to have make information up here.

#include <GL/glut.h>
#include <stdlib.h>
#include <iostream>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <signal.h>

//#include <thrust/system_error.h>
//#include <thrust/system/cuda/error.h>
//#include <sstream>

#define PI 3.141592653589793238

//All includes which require glib
//This section is required because of a noinline conflict with nvcc
#undef __noinline__
#include <gtk/gtk.h>
#define __noinline__ __attribute__((noinline))

using namespace std;

float *pixelsJulia_CPU, *pixelsMandel_CPU, *pixels_CPU;

float *pixelsJulia_GPU, *pixelsMandel_GPU, *pixels_GPU;

//dim3 dimBlock;

//See coordinates.pdf for explanation of this coordinate system.
float m[] = {0, 0.75, 0, 0};
float c[] = {-2, -2, 0, 0};
float v[] = {4, 0, 0, 0};
float o[] = {0, 0, 0, 0};
float coordinates[] = {0,0};
//See coordinates.pdf for explanation of this coordinate system.

//GPU vectors for the coordinate system (to be copied later)
float *m_gpu, *c_gpu, *v_gpu, *o_gpu;

float t = 0;
float tmod = 100.0;
float titer = 1.0;
float moveiter = 0.001;

float viter = 0.001;

int N = 100;
int mandelID; 
int juliaID; 
unsigned int window_height = 960;
unsigned int window_width = 2*window_height;

int framecount = 0;
/*
float xMin = -2.0;
float xMax =  2.0;
float yMin = -2.0;
float yMax =  2.0;

float stepSizeX = (xMax - xMin)/((float)window_width);
float stepSizeY = (yMax - yMin)/((float)window_height);
*/

void ortho_coordinates(){

    //Given the view corner, view vector, and lat/long,
    //  return the 4 coordinates of the orthogonal vector.
    
    //Create a basis for the orthogonal complement of v.
    float e0[] = {-v[1], v[0],   0,   0};
    float e1[] = {-v[2],   0, v[0],   0};
    float e2[] = {-v[3],   0,   0, v[0]};
    
    //Length of v
    float R = sqrt(v[0]*v[0]+ v[1]*v[1]+ v[2]*v[2]+ v[3]*v[3]);
    
    //x,y,z coordinates given our lat/long coordinates
    float x = R*cos(coordinates[0])*cos(coordinates[1]);
    float y = R*cos(coordinates[0])*sin(coordinates[1]);
    float z = R*sin(coordinates[0]);
    
    //Transform the coordinates from the basis we just created to the overall space. 
    o[0] = x*e0[0] + y*e1[0] + z*e2[0];
    o[1] = x*e0[1] + y*e1[1] + z*e2[1];
    o[2] = x*e0[2] + y*e1[2] + z*e2[2];
    o[3] = x*e0[3] + y*e1[3] + z*e2[3];
    
    //length of o
    float r = sqrt(o[0]*o[0] + o[1]*o[1] + o[2]*o[2] + o[3]*o[3]);

    //Normalize o to the length of v.
    o[0] = R*o[0]/r;
    o[1] = R*o[1]/r;
    o[2] = R*o[2]/r;
    o[3] = R*o[3]/r;
}

static void
print_hello (GtkWidget *widget,
             gpointer   data)
{
  g_print ("Hello World\n");
}

static gboolean
on_delete_event (GtkWidget *widget,
                 GdkEvent  *event,
                 gpointer   data)
{
  /* If you return FALSE in the "delete_event" signal handler,
   * GTK will emit the "destroy" signal. Returning TRUE means
   * you don't want the window to be destroyed.
   *
   * This is useful for popping up 'are you sure you want to quit?'
   * type dialogs.
   */

  g_print ("delete event occurred\n");

  return TRUE;
}

void AllocateMemory(){
	cudaMalloc((void**)&pixelsJulia_GPU, window_width/2*window_height*3*sizeof(float));
	pixelsJulia_CPU = (float *)malloc(window_width/2*window_height*3*sizeof(float));
	cudaMalloc((void**)&pixelsMandel_GPU, window_width/2*window_height*3*sizeof(float));
	pixelsMandel_CPU = (float *)malloc(window_width/2*window_height*3*sizeof(float));
	cudaMalloc((void**)&pixels_GPU, window_width*window_height*3*sizeof(float));
	pixels_CPU = (float *)malloc(window_width*window_height*3*sizeof(float));
	cudaMalloc((void**)&m_gpu, 4*sizeof(float));
	cudaMalloc((void**)&c_gpu, 4*sizeof(float));
	cudaMalloc((void**)&v_gpu, 4*sizeof(float));
	cudaMalloc((void**)&o_gpu, 4*sizeof(float));

} // */	//Saves the appropriate memory chunks for later use.
	//References the globally defined variables.


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


__global__ void cudaColorJulia(float *pixelsJulia_GPU, float *m, float *c, float *v, float *o){
	
	//j = {1, i, j, k}
	//Start from the value for the first corner, and add values from v and o
	//  based on grid position.
	float j[] = {
	    c[0] + (((float)threadIdx.x)/(blockDim.x))*v[0] + (((float)blockIdx.x)/(gridDim.x))*o[0],
	    c[1] + (((float)threadIdx.x)/(blockDim.x))*v[1] + (((float)blockIdx.x)/(gridDim.x))*o[1],
	    c[2] + (((float)threadIdx.x)/(blockDim.x))*v[2] + (((float)blockIdx.x)/(gridDim.x))*o[2],
	    c[3] + (((float)threadIdx.x)/(blockDim.x))*v[3] + (((float)blockIdx.x)/(gridDim.x))*o[3],
	};
	
	float mag,maxMag, t;
	int maxCount = 200;
	int count = 0;
	maxMag = 10;
	mag = 0.0;

	while (mag < maxMag && count < maxCount) 
	{
		t = j[0];
			
		j[0] = j[0]*j[0] - j[1]*j[1] - j[2]*j[2] - j[3]*j[3] + m[0];
		j[1] = 2*t*j[1]+m[1];
		j[2] = 2*t*j[2]+m[2];
		j[3] = 2*t*j[3]+m[3];
		mag = sqrt(j[0]*j[0] + j[1]*j[1] + j[2]*j[2] + j[3]*j[3]);
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

void processSpecialKeys(int key, int x, int y) {

	switch(key) {
		case GLUT_KEY_RIGHT :
			m[0] = m[0] + moveiter;	break;
		case GLUT_KEY_LEFT :
			m[0] = m[0] - moveiter;	break;
		case GLUT_KEY_UP :
			m[1] = m[1] + moveiter;	break;
		case GLUT_KEY_DOWN :
			m[1] = m[1] - moveiter;	break;
	}
	
}

void view_zoom_in() {

    //The corner moves slightly in the direction of v and o.
    //Then v becomes smaller and o is recalculated.
    
    c[0] = c[0] + viter * v[0] + viter * o[0];
    c[1] = c[1] + viter * v[1] + viter * o[1];
    c[2] = c[2] + viter * v[2] + viter * o[2];
    c[3] = c[3] + viter * v[3] + viter * o[3];
    
    v[0] = v[0] - 2*viter*v[0];
    v[1] = v[1] - 2*viter*v[1];
    v[2] = v[2] - 2*viter*v[2];
    v[3] = v[3] - 2*viter*v[3];
        
    ortho_coordinates();
    
}

void view_zoom_out() {
    
    //The corner moves slightly in the OPPOSITE direction of v and o.
    //Then v becomes larger and o is recalculated.
    
    c[0] = c[0] - viter * v[0] - viter * o[0];
    c[1] = c[1] - viter * v[1] - viter * o[1];
    c[2] = c[2] - viter * v[2] - viter * o[2];
    c[3] = c[3] - viter * v[3] - viter * o[3];
    
    v[0] = v[0] + 2*viter*v[0];
    v[1] = v[1] + 2*viter*v[1];
    v[2] = v[2] + 2*viter*v[2];
    v[3] = v[3] + 2*viter*v[3];
        
    ortho_coordinates();
    
}

void processNormalKeys(unsigned char key, int x, int y) {
    
    switch(key) {
        case 49: // 1
        case 33: // !
		    c[0] = c[0] + viter;   break;
        case 50: // 2
        case 64: // @
		    c[1] = c[1] + viter;   break;
        case 51: // 3
        case 35: // #
		    c[2] = c[2] + viter;   break;
        case 52: // 4
        case 36: // $
		    c[3] = c[3] + viter;   break;
        case 53: // 5
        case 37: // %
		    v[0] = v[0] + viter;   break;
        case 54: // 6
        case 94: // ^
		    v[1] = v[1] + viter;   break;
        case 55: // 7
        case 38: // &
		    v[2] = v[2] + viter;   break;
        case 56: // 8
        case 42: // *
		    v[3] = v[3] + viter;   break;
        case 57: // 9
        case 40: // (
		    coordinates[0] = min(coordinates[0] + viter, PI/2);   break;
        case 48: // 0
        case 41: // )
            coordinates[1] = coordinates[1] + viter > 2*PI ? 0 : coordinates[1] + viter;   break;
        case 45: // -
        case 95: // _
		    viter = viter * 1.2;  break; 
        case 61: // =
        case 43: // +
		    view_zoom_in();   break;
        case 81: // Q
        case 113: // q
		    c[0] = c[0] - viter;   break; 
        case 87: // W
        case 119: // w
		    c[1] = c[1] - viter;   break; 
        case 69: // E
        case 101: // e
		    c[2] = c[2] - viter;   break; 
        case 82: // R
        case 114: // r
		    c[3] = c[3] - viter;   break; 
        case 84: // T
        case 116: // t
		    v[0] = v[0] - viter;   break; 
        case 89: // Y
        case 121: // y
		    v[1] = v[1] - viter;   break; 
        case 85: // U
        case 117: // u
		    v[2] = v[2] - viter;   break; 
        case 73: // I
        case 105: // i
		    v[3] = v[3] - viter;   break; 
        case 79: // O
        case 111: // o
		    coordinates[0] = max(coordinates[0] - viter, -PI/2);   break;
        case 80: // P
        case 112: // p
            coordinates[1] = coordinates[1] - viter < 0 ? 2*PI : coordinates[1] - viter;   break;
        case 91: // [
        case 123: // {
		    viter = viter / 1.2;  break; 
        case 93: // ]
        case 125: // }
		    view_zoom_out();   break; 
        case 65: // A
        case 97: // a
		    m[0] = m[0] + moveiter;   break; 
        case 83: // S
        case 115: // s
		    m[1] = m[1] + moveiter;   break; 
        case 68: // D
        case 100: // d
		    m[2] = m[2] + moveiter;   break; 
        case 70: // F
        case 102: // f
		    m[3] = m[3] + moveiter;   break; 
        case 90: // Z
        case 122: // z
		    m[0] = m[0] - moveiter;   break; 
        case 88: // X
        case 120: // x
		    m[1] = m[1] - moveiter;   break; 
        case 67: // C
        case 99: // c
		    m[2] = m[2] - moveiter;   break; 
        case 86: // V
        case 118: // v
		    m[3] = m[3] - moveiter;   break; 
        case 71: // G
        case 103: // g
		    moveiter = moveiter * 1.2;  break;
        case 66: // B
        case 98: // b
		    moveiter = moveiter/1.2;    break;
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
			m[0] = ((float)x)/window_width*8.0-6.5;
			m[1] = (-(float)y)/window_height*4.0+2.0;	break;
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

/*void throw_on_cuda_error(cudaError_t code, const char *file, int line)
{
  if(code != cudaSuccess)
  {
    std::stringstream ss;
    ss << file << "(" << line << ")";
    std::string file_and_line;
    ss >> file_and_line;
    throw thrust::system_error(code, thrust::cuda_category(), file_and_line);
  }
}*/

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void display(void){

    framecount = framecount + 1;
    
    ortho_coordinates();
    
    if (framecount%100 == 0){
        printf("frame: %d \n", framecount);
        printf("m: %f, %f *i, %f *j, %f *k \n" , m[0], m[1], m[2], m[3]);
        printf("c: %f, %f *i, %f *j, %f *k \n" , c[0], c[1], c[2], c[3]);
        printf("v: %f, %f *i, %f *j, %f *k \n" , v[0], v[1], v[2], v[3]);
        printf("o: %f, %f *i, %f *j, %f *k \n" , o[0], o[1], o[2], o[3]);
    };
    cudaMemcpy(m_gpu, m, 4*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(c_gpu, c, 4*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(v_gpu, v, 4*sizeof(float), cudaMemcpyHostToDevice);
    gpuErrchk(cudaMemcpy(o_gpu, o, 4*sizeof(float), cudaMemcpyHostToDevice));
    
    
	cudaColorJulia<<<window_width/2, window_height>>>(pixelsJulia_GPU, m_gpu, c_gpu, v_gpu, o_gpu);
	cudaMemcpy(pixelsJulia_CPU, pixelsJulia_GPU,
		window_width/2*window_height*3*sizeof(float),
		cudaMemcpyDeviceToHost);
	cudaColorMandelbrot<<<window_width/2, window_height>>>(pixelsMandel_GPU, m[0], m[1]);
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

