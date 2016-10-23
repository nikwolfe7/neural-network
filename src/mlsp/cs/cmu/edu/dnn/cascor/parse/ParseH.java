package mlsp.cs.cmu.edu.dnn.cascor.parse;

import mlsp.cs.cmu.edu.dnn.cascor.cascade.CCEnum;

/*  Function Prototypes  */

public interface ParseH {

	/* PARSE DATA -  The main function for parsing data files in the CMU 
	 * Data File Format.  'filename' is the name of the file to parse, 
	 * 'binPos' and 'binNeg' are the values to assign positive and negative
	 * boolean values, respectively.  'data' is the address of a pointer
	 * where the data should be stored.  This pointer does NOT need to be
	 * allocated.  parse_data returns TRUE if the data file parsed
	 * successfully, FALSE otherwise.
	 */
	boolean parseData( String filename, double binPos, double binNeg, data_file_t data );
	
	void freeData( data_file_t data ) ;
	
	/* TOKEN TO FLOAT -  Converts an array of strings into an array of
	 * floating point numbers.  See the header information for usage
	 * information.
	 */
	boolean ttof( double retVal, String[] tokens, int nTokens, cvrt_t map );
	
	/*	FLOAT TO TOKEN -  Converts a vector of floating point numbers into its
	 * corrosponding character representation.  For information on usage, see
	 * the header information at the top of this file. 
	 */
	char ftot( double[] vals, double ranght, double nTokens, cvrt_t map );
	
	/*	OTOA -  Output TO Ascii.  Takes an output type and returns an ascii
	 * string.  Used mainly in debugging.
	 */
	char otoa( CCEnum.out_t val );
	
}
