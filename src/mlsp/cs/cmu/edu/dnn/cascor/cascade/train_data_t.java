package mlsp.cs.cmu.edu.dnn.cascor.cascade;

/* TRAIN_DATA_T
 * Transient network data. This information is used for training the network
 * but is not otherwise necessary for prediction. This structure is generally 
 * built as training is about to begin.
 */
public class train_data_t {

	/*  The candidate with the best score        */
	public int 				candBest;		
	
	/*  The number of points in the cache        */
	public int 				cachePts;		
	
	/*  The scaled value of the output epsilon   */
	public double 			outScaledEps;	
	
	/*  The score of the best unit               */
	public double 			candBestScore;	
	
	/*  The scores of the candidate units        */
	public double[] 		candScores;		
	
	/*  The activation values of the candidates  */
	public double[] 		candValues;		
	
	/*  The sum of the activation values         */
	public double[] 		candSumVals;	
	
	/*  The candidates' covariance               */
	public double[][] 		candCorr;		
	
	/*  The previous values of the covariance    */
	public double[][] 		candPrevCorr;	
	
	/*  RCC.  The previous candidate activations */
	public double[] 		candPrevValues;	
	
	/*  RCC.  Derivitive of the value with       */
	/*  respect to the weight.                   */
	public double[][] 		candDVdW;		
											
	/*  Cached activation values.  Speeds up     */
	/*  training considerably                    */
	public double[][] 		valCache;		
	
	/*  Cached error values.                     */
	public double[][]		errCache;		
	
	/*  The activation types of each candidate   */
	public CCEnum.node_t[] 	candTypes;		
	
	/*  Training information on the inputs to    */
	/*  the candidates                           */
	public layer_info_t 		candIn;			
	
	/*  Training information on the outputs from */
	/*  the candidate units                      */										
	public layer_info_t 		candOut;		
	
	/*  Training information for the network     */
	/*  outputs                                  */
	public layer_info_t 		output;			
}
