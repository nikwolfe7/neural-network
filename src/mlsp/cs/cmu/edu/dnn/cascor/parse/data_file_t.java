package mlsp.cs.cmu.edu.dnn.cascor.parse;

import mlsp.cs.cmu.edu.dnn.cascor.cascade.CCEnum;

/* DATA SET OBJECT - This structure contains (part of) the information needed to 
 * train a network to perform a specific task. The filename that the data came
 * from, the output types, number of inputs, outputs and the actual 
 * training vectors.  
 */
public class data_file_t {
	
	public String filename;
	public CCEnum.out_t outputType;
	public int nInputs;
	public int nOutputs;
	public int nInNodes;
	public int nOutNodes;
	public int nDataSets;
	public double binPos;
	public double binNeg;
	public cvrt_t inputMap;
	public cvrt_t outputMap;
	public data_set_t dataSets;
	public data_set_t train;
	public data_set_t validate;
	public data_set_t test;
	public data_set_t predict;

}
