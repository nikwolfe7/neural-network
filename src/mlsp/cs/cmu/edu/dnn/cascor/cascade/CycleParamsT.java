package mlsp.cs.cmu.edu.dnn.cascor.cascade;

/* CYCLE_PARAMS_T
 * These parameters are specific to a specific part of the training cycle.
 * Therefore, one of these structures exist for both the output training
 * and candidate training phases.                                           
 */
public class CycleParamsT {
	
	public int epochs;				/* The number of training epochs to perform in      */
    								/* each phase before a TIMEOUT is declared.  In     */
									/* general, a TIMEOUT should never be declared.     */
	public int patience; 			/* The number of epochs without significant change  */
									/* before the training is declared STAGNANT         */
	public double changeThreshold;	/*  The relative size of change required to be      */
    								/* considered 'significant'                         */
}
