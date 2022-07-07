#include "mex.h"
#include "computeLDOF.cpp"

/*************************************************************/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    
    if (nrhs < 2 || nrhs > 6 ) mexErrMsgTxt("INPUT: (image1, image2, sigma, alpha, beta, gamma) ");
    if (nlhs != 1) mexErrMsgTxt("OUTPUT: optical_flow ");
    
    if (!mxIsDouble(prhs[0]) || mxGetNumberOfDimensions(prhs[0]) != 3) mexErrMsgTxt("First input must be a 3-dimensional array of noncomplex doubles.");
    if (!mxIsDouble(prhs[1]) || mxGetNumberOfDimensions(prhs[0]) != 3) mexErrMsgTxt("Second input must be a 3-dimensional array of noncomplex doubles.");
    
    double* image1 = mxGetPr(prhs[0]);
    double* image2 = mxGetPr(prhs[1]);
    
    const int *dims1, *dims2;
    
    dims1 = mxGetDimensions(prhs[0]);
    dims2 = mxGetDimensions(prhs[1]);
    
    if ((dims1[0] != dims2[0]) || (dims1[1] != dims2[1]) || (dims1[2] != dims2[2]))
        mexErrMsgTxt("Input images must have the same size !");
        
    CTensor<float> aImage1(dims1[1], dims1[0], dims1[2]);
    CTensor<float> aImage2(dims2[1], dims2[0], dims2[2]);
    
	  int i = 0;
	  for (int z = 0; z < aImage1.zSize(); z++)
	    for (int x = 0; x < aImage1.xSize(); x++)
		    for (int y = 0; y < aImage1.ySize(); y++,i++) {
          aImage1(x,y,z) = image1[i];
          aImage2(x,y,z) = image2[i];
        }
  
    float sigma = 0.8f;
    float alpha = 30; 
    float beta = 300;
    float gamma = 5;
       
    if (nrhs >= 3) sigma = (float)mxGetScalar(prhs[2]);
    if (nrhs >= 4) alpha = (float)mxGetScalar(prhs[3]);
    if (nrhs >= 5) beta = (float)mxGetScalar(prhs[4]);
    if (nrhs >= 6) gamma = (float)mxGetScalar(prhs[5]);
    
    
    CTensor<float> aFlow;
    runFlow(aImage1,aImage2,aFlow,sigma,alpha,beta,gamma);
    
    int *dims_of = new int[3];
    dims_of[0] = dims1[0]; dims_of[1] = dims1[1]; dims_of[2] = 2;
    
    int ndims = 3;
    plhs[0] = mxCreateNumericArray(ndims, dims_of, mxDOUBLE_CLASS, mxREAL);
    double *optical_flow = mxGetPr(plhs[0]);
    
    i = 0;
	  for (int z = 0; z < aFlow.zSize(); z++)
	    for (int x = 0; x < aFlow.xSize(); x++)
	      for (int y = 0; y < aFlow.ySize(); y++,i++)
          optical_flow[i] = aFlow(x,y,z);
    
    delete[] dims_of;
    
}
