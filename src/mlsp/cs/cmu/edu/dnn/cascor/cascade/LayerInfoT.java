package mlsp.cs.cmu.edu.dnn.cascor.cascade;

/* LAYER_INFO_T
 * Contains training data for a single layer of the network.  
 * Instances are constructed for the output, candidate input and 
 * candidate output layers. 
 */
public class LayerInfoT {
	
	public double shrinkFactor;		/*  This is related to mu.  See [1]                   */
	public double[][] weights;		/*  Only used for candidate layers                    */
	public double[][] deltas;	    /*  The previous weight changes                       */
	public double[][] slopes;		/*  The slope of the error function at this point     */
	public double[][] pSlopes;		/*  The previous value of the slope at this point     */

}
