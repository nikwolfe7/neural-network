package mlsp.cs.cmu.edu.dnn.cascor.cascade;

/* ERROR_DATA_T 
 * Contains error information on network performance
 */
public class ErrorDataT {
	
	/*  Number of incorrect bits  					*/
	public int bits;			
	
	/*  Computed error index  						*/
	public double index;		
	
	/*  The Sum of the Square Differences  			*/
	public double sumSqDiffs;	
	
	/*  The Sum of the Square Errors (using EPrime)	*/
	public double sumSqError;	
	
	/*  The error at each output  					*/
	public double[] errors;		
	
	/*  Temporary error vector in case of no cache	*/
	public double[] tempErrors; 
	
	/*  The sum of the error at each output  		*/
	public double[] sumError; 	

}
