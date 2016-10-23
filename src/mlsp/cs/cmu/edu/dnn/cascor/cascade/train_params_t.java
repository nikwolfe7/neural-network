package mlsp.cs.cmu.edu.dnn.cascor.cascade;

/* TRAIN_PARM_T
 * This is the main structure used to contain training parameters.  
 * All that is necessary for network training is contained herein.                   
 */
public class train_params_t {
	
	/* The maximum number of units to add   */
	/* to the network being trained         */
	public int maxNewUnits;			
	
	/* The number of training cycles to     */
	/* perform without improvement in       */
	/* cross-validation generalization      */
	public int validationPatience;	 	
    			
	/* Number of candidates in the          */
	/* training pool                        */
	public int nCand;					
	
	/* Amount to offset the error prime     */
	/* when training outputs.  See [1]      */
	/* for details of why this helps        */
	public double outPrimeOffset;	 		
	
	/* The maximum variance of random       */
	/* weights from zero                    */
	public double weightRange;			
	
	/* The maximum Error Index that is      */
	/* considered a victory                 */
	public double indexThreshold;	 	
	
	/* The maximum fractional variance of   */
	/* a unit from its goal to be           */
	/* considered correct                   */
	public double scoreThreshold;	 
	
	/* Maximum value of VARSIGMOID units    */
	public double sigMax;			 		
	
	/* Minimum value of VARSIGMOID units    */
	public double sigMin; 			 	
	
	/* Ok to overshoot the desired goal?    */
	public boolean overshootOK;		 	
	
	/* Is value and error cache in use?     */
	public boolean useCache;		 		
	
	/* Test the network after training?     */
	public boolean test;			 	
	
	/* Cross-validate the network during    */
	/* training?                            */
	public boolean validate;				
    								 				
	/* Train a recurrent network?           */
	public boolean recurrent;		 		
	
	/* Type of candidate to comprise pool   */
	public CCEnum.node_t candType;	 			
	
	/* Network architecture to use          */
	public CCEnum.algo_t algorithm;	 			
	
	/* Measure that determines success      */
	public CCEnum.error_t errorMeasure;			
	
	/* Parameters for candidates inputs     */
	public update_params_t candInUpdate;		   
	
	/* Parameters for candidates outputs    */
	public update_params_t candOutUpdate;			
	
	/* Parameters for network outputs       */
	public update_params_t outputUpdate;			
	
	/* Candidate phase parameters           */
	public cycle_params_t candidateParams;		
	
	/* Output phase parameters              */
	public cycle_params_t outputParams;			
	
}
