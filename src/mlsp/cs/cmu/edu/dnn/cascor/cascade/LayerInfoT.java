package mlsp.cs.cmu.edu.dnn.cascor.cascade;

/* LAYER_INFO_T
 * Contains training data for a single layer of the network.  
 * Instances are constructed for the output, candidate input and 
 * candidate output layers. 
 */
public class LayerInfoT {
	
	/*  This is related to mu.  See [1]                   */
	public double shrinkFactor;		
	
	/*  Only used for candidate layers                    */
	public double[][] weights;		
	
	/*  The previous weight changes                       */
	public double[][] deltas;	   
	
	/*  The slope of the error function at this point     */
	public double[][] slopes;		
	
	/*  The previous value of the slope at this point     */
	public double[][] pSlopes;		

}
