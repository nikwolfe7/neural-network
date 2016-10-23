package mlsp.cs.cmu.edu.dnn.cascor.cascade;

/* Enums taken from cascade.h */

public final class CCEnum {
	
	/*  Node types  */
	public static enum node_t {
		UNDEFINED,
		VARIED,
		SIGMOID,
		ASIGMOID,
		VARSIGMOID,
		GAUSSIAN,
		LINEAR
	}
	
	/*  Training algorithms  */
	public static enum algo_t {
		CASCOR,
		CASCADE2
	} 
	
	/*  Method of determining network error  */
	public static enum error_t {
		INDEX,
		BITS
	}

	/*  Training statuses  */
	public static enum status_t {
		TRAINING,
		TIMEOUT,
		STAGNANT,
		WIN,
		LOSS
	}	
	
	/*  Type Declarations  */
	/*  Outputs can be either BINARY or					 	*/
	/*  CONTinuous.  BINARY values are used for binary		*/
	/*  and enumerated units while CONTinuous values are 	*/
	/*  used for standard floating point numbers.        	*/
	public static enum out_t {
		CONT,					
		BINARY					
	}
	
	/* PARM_VAR_T
	 * These are enumerations of the various types of data stored in the table
	 * located in 'interface.c'.  These enumerations help the data be interpreted
	 * correctly.                                                               
	 */
	public static enum param_var_t {
		INT,	  /* Integer value */	
		BOOLEAN,  /*  Boolean value                                   */
	    NODE,     /*  Node type (i.e. Sigmoid, Gaussian, etc.)        */
	    ALGO,     /*  Algorithm type (Cascor/Cascade-2)               */
	    ERR,      /*  Error type (Bits/Index)                         */
	    FUNC      /*  A function's address                            */
	}

}
