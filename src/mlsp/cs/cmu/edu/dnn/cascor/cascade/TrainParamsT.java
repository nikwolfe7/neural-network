package mlsp.cs.cmu.edu.dnn.cascor.cascade;

/* TRAIN_PARM_T
 * This is the main structure used to contain training parameters.  
 * All that is necessary for network training is contained herein.                   
 */
public class TrainParamsT {
	
	public int 				maxNewUnits;			/* The maximum number of units to add   */
									 				/* to the network being trained         */
	public int 				validationPatience;	 	/* The number of training cycles to     */
    								 				/* perform without improvement in       */
    								 				/* cross-validation generalization      */
	public int 				nCand;					/* Number of candidates in the          */
    								 				/* training pool                        */
	public double 			outPrimeOffset;	 		/* Amount to offset the error prime     */
    								 				/* when training outputs.  See [1]      */
    								 				/* for details of why this helps        */
	public double 			weightRange;			/* The maximum variance of random       */
    								 				/* weights from zero                    */
	public double 			indexThreshold;	 		/* The maximum Error Index that is      */
    								 				/* considered a victory                 */
	public double 			scoreThreshold;	 		/* The maximum fractional variance of   */
    								 				/* a unit from its goal to be           */
    								 				/* considered correct                   */
	public double 			sigMax;			 		/* Maximum value of VARSIGMOID units    */
	public double 			sigMin; 			 	/* Minimum value of VARSIGMOID units    */
	public boolean 			overshootOK;		 	/* Ok to overshoot the desired goal?    */
	public boolean 			useCache;		 		/* Is value and error cache in use?     */
	public boolean 			test;			 		/* Test the network after training?     */
	public boolean 			validate;				/* Cross-validate the network during    */
    								 				/* training?                            */
	public boolean 			recurrent;		 		/* Train a recurrent network?           */
	public CCEnum.NodeT 	candType;	 			/* Type of candidate to comprise pool   */
	public CCEnum.AlgoT 	algorithm;	 			/* Network architecture to use          */
	public CCEnum.ErrorT 	errorMeasure;			/* Measure that determines success      */
	public UpdateParamsT 	candInUpdate;		    /* Parameters for candidates inputs     */
	public UpdateParamsT	candOutUpdate;			/* Parameters for candidates outputs    */
	public UpdateParamsT	outputUpdate;			/* Parameters for network outputs       */
	public CycleParamsT		candidateParams;		/* Candidate phase parameters           */
	public CycleParamsT		outputParams;			/* Output phase parameters              */
	
}
