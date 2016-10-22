package mlsp.cs.cmu.edu.dnn.cascor.cascade;

/* TRAIN_DATA_T
 * Transient network data. This information is used for training the network
 * but is not otherwise necessary for prediction. This structure is generally 
 * built as training is about to begin.
 */
public class TrainDataT {

	public int 				candBest;		/*  The candidate with the best score        */
	public int 				cachePts;		/*  The number of points in the cache        */
	public double 			outScaledEps;	/*  The scaled value of the output epsilon   */
	public double 			candBestScore;	/*  The score of the best unit               */
	public double[] 		candScores;		/*  The scores of the candidate units        */
	public double[] 		candValues;		/*  The activation values of the candidates  */
	public double[] 		candSumVals;	/*  The sum of the activation values         */
	public double[][] 		candCorr;		/*  The candidates' covariance               */
	public double[][] 		candPrevCorr;	/*  The previous values of the covariance    */
	public double[] 		candPrevValues;	/*  RCC.  The previous candidate activations */
	public double[][] 		candDVdW;		/*  RCC.  Derivitive of the value with       */
											/*  respect to the weight.                   */
	public double[][] 		valCache;		/*  Cached activation values.  Speeds up     */
    										/*  training considerably                    */
	public double[][]		errCache;		/*  Cached error values.                     */
	public CCEnum.NodeT[] 	candTypes;		/*  The activation types of each candidate   */
	public LayerInfoT 		candIn;			/*  Training information on the inputs to    */
											/*  the candidates                           */
	public LayerInfoT 		candOut;		/*  Training information on the outputs from */
    										/*  the candidate units                      */
	public LayerInfoT 		output;			/*  Training information for the network     */
    										/*  outputs                                  */
}
