package mlsp.cs.cmu.edu.dnn.cascor.cascade;

/* Enums taken from cascade.h */

public final class CCEnum {
	
	/*  Node types  */
	public static enum NodeT {
		UNDEFINED,
		VARIED,
		SIGMOID,
		ASIGMOID,
		VARSIGMOID,
		GAUSSIAN,
		LINEAR
	}
	
	/*  Training algorithms  */
	public static enum AlgoT {
		CASCOR,
		CASCADE2
	} 
	
	/*  Method of determining network error  */
	public static enum ErrorT {
		INDEX,
		BITS
	}

	/*  Training statuses  */
	public static enum StatusT {
		TRAINING,
		TIMEOUT,
		STAGNANT,
		WIN,
		LOSS
	}	
	
	/*  Type Declarations  */
	public static enum OutT {	/*  Outputs can be either BINARY or                  */
		CONT,					/*  CONTinuous.  BINARY values are used for binary   */
		BINARY					/*  and enumerated units while CONTinuous values are */
								/*  used for standard floating point numbers.        */					
	}

}
