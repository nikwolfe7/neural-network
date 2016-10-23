package mlsp.cs.cmu.edu.dnn.cascor.cascade;

import mlsp.cs.cmu.edu.dnn.cascor.parse.cvrt_t;

/* NET_T
 * This is the network data structure.  This structure contains all necessary
 * information about the network.  Using the information contained within this
 * structure, feedforward prediction is possible.                           
 */
public class net_t {
	
	/* Name of the network                    */
	public String name;			 	
	
	/* Filename the network is stored in      */
	public String fileName;		 	
	
	/* Description of the network. Not used.  */
	public String descript; 	 		
	
	/* Total number of epochs this network    */
	/* has been trained                       */
	public int epochsTrained;	 	
	
	/* Number of units in the network.  This  */
	/* includes inputs, hidden units and the  */
	/* bias unit, but not the outputs         */
	public int nUnits;			 	
								 			
	/* Number of input units to the network   */
	public int nInputs;			
	
	/* Number of ouputs from the network      */
	public int nOutputs;		  	
	
	/* Current number of hidden units         */
	public int nHiddenUnits;	  	
	
	/* Maximum number of hidden units that    */
	public int maxNewUnits;		
	
	/* Unit activation values                 */
	public double[] values;		  		
	
	/* Temp float vector to be used when the  */
	/* cache is not in use                    */
	public double[] tempValues;   		
	
	/* Interior weights. Weights to outputs   */
	/* not included.                          */
	public double[][] weights;	  		
	
	/* Activation levels of the outputs       */
	public double[] outValues;    		 
	
	/* Weights to the outputs                 */
	public double[][] outWeights;   		
    
	/* Maximum value of a VARSIGMOID          */
	public double sigmoidMax;     	
    
	/* Minimum value of a VARSIGMOID          */
	public double sigmoidMin;     	
    
	/* Is this net recurrent?                 */
	public boolean recurrent;      	
    
	/* Map from tokens to raw inputs          */
	public cvrt_t inputMap;
    
	/* Maps from raw outputs to tokens        */
	public cvrt_t outputMap;				
    
	/* Types for the interior units           */
	public CCEnum.node_t[] unitTypes;		
    
	/* Types of the outputs                   */
	public CCEnum.node_t[] outputTypes;	
    
	/* Pointer to the next layer... 		  */
	public net_t next;			
	
}
