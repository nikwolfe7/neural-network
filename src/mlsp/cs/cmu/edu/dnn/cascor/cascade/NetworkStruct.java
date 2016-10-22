package mlsp.cs.cmu.edu.dnn.cascor.cascade;

/* NET_T
 * This is the network data structure.  This structure contains all necessary
 * information about the network.  Using the information contained within this
 * structure, feedforward prediction is possible.                           
 */
public class NetworkStruct {

	public String 		name;			 	/* Name of the network                    */
	public String 		fileName;		 	/* Filename the network is stored in      */
	public String 		descript; 	 		/* Description of the network. Not used.  */
	public int 			epochsTrained;	 	/* Total number of epochs this network    */
								 			/* has been trained                       */
	public int 			nUnits;			 	/* Number of units in the network.  This  */
    							 			/* includes inputs, hidden units and the  */
    							 			/* bias unit, but not the outputs         */
	public int 			nInputs;			/* Number of input units to the network   */
	public int 			nOutputs;		  	/* Number of ouputs from the network      */
	public int 			nHiddenUnits;	  	/* Current number of hidden units         */
	public int 			maxNewUnits;		/* Maximum number of hidden units that    */
	public double[] 	values;		  		/* Unit activation values                 */
	public double[] 	tempValues;   		/* Temp float vector to be used when the  */
    							  			/* cache is not in use                    */
	public double[][] 	weights;	  		/* Interior weights.  Weights to outputs  */
    							  			/* not included.                          */
	public double[] 	outValues;    		/* Activation levels of the outputs       */
    public double[][] 	outWeights;   		/* Weights to the outputs                 */
    public double 		sigmoidMax;     	/* Maximum value of a VARSIGMOID          */
    public double 		sigmoidMin;     	/* Minimum value of a VARSIGMOID          */
    public boolean 		recurrent;      	/* Is this net recurrent?                 */
    //cvrt_t          *inputMap,      		/* Map from tokens to raw inputs          */
    //*outputMap;    	 					/* Maps from raw outputs to tokens        */
    
    public CCEnum.NodeT[] 	unitTypes;		/* Types for the interior units           */
    public CCEnum.NodeT[]   outputTypes;	/* Types of the outputs                   */
    public NetworkStruct	next;			/* Pointer to the next layer... 		  */
	
}
