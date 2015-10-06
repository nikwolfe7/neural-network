package mlsp.cs.cmu.edu.structure;

import mlsp.cs.cmu.edu.elements.Edge;

public class WeightMatrix extends NetworkElementLayer {

	private Edge[] weightMatrix;
	private int rows, cols;
	
	public WeightMatrix(int rows, int cols, Edge... elements) {
		super(elements);
		this.weightMatrix = elements;
		this.rows = rows;
		this.cols = cols;
	}
	
	public Edge getEdge(int row, int col) {
		return weightMatrix[row * cols + col];
	}

}
