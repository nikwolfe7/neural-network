package mlsp.cs.cmu.edu.dnn.cascor.cascade;

/* UPDATE_PARAMS_T
 * These are parameters that are associated with the weight updates of a
 * specific layer.  Accordingly, instances of this structure exist for the
 * output, candidate in and candidate out layers.                           
 */
public class UpdateParamsT {

	/* Learning rate parameter.  Higher rates can decrease    */
	/* training time, but may cause learning to go unstable   */
	public double epsilon;		
	
	/* Maximum step size parameter as described by Fahlman    */
	/* in the Quickprop paper [1].  Usually not worth tuning  */
	public double mu;			
	
	/* Weight decay.  Causes weights to decay towards zero.   */
	/* If you get monstrous weights, set this to ~0.0001 or   */
	/* less (it doesn't take much).                           */
	public double decay;		
}
