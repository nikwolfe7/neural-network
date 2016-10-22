package mlsp.cs.cmu.edu.dnn.cascor.cascade;

/* ERROR_DATA_T 
 * Contains error information on network performance
 */
public class ErrorDataT {
	
	public int bits;			/*  Number of incorrect bits  					*/
	public double index;		/*  Computed error index  						*/
	public double sumSqDiffs;	/*  The Sum of the Square Differences  			*/
	public double sumSqError;	/*  The Sum of the Square Errors (using EPrime)	*/
	public double[] errors;		/*  The error at each output  					*/
	public double[] tempErrors; /*  Temporary error vector in case of no cache	*/
	public double[] sumError; 	/*  The sum of the error at each output  		*/

}
