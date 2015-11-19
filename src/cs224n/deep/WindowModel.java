package cs224n.deep;
import java.util.*;

import org.ejml.data.*;
import org.ejml.simple.*;
import java.text.*;

public class WindowModel {
	
	private HashMap<String, String> baselineWordMap;
	protected SimpleMatrix L, W, U;
	public int windowSize,wordSize, hiddenSize;

	public WindowModel(int _windowSize, int _hiddenSize, double _lr){
		baselineWordMap = new HashMap<String, String>();
	}

	/**
	 * Initializes the weights randomly. 
	 */
	public void initWeights(){
		// TODO Add one more column for b
		int H = hiddenSize; //fanOut = H
		int Cn = windowSize * FeatureFactory.allVecs.numRows(); //fanIn = nC
		int K = Datum.POSSIBLE_LABELS.length;
		double range = Math.sqrt(6) / Math.sqrt(H + Cn);
		W = SimpleMatrix.random(H, Cn, -range, range, new Random());
		U = SimpleMatrix.random(K, H, -range, range, new Random());
	}


	/**
	 * Simplest SGD training 
	 */
	public void train(List<Datum> trainData ){
		this.baselineTrain(trainData);
	}


	public void test(List<Datum> testData){
		this.baselineTest(testData);
	}
	
	private void baselineTrain(List<Datum> trainData){
		for (Datum datum : trainData)
		{
			this.baselineWordMap.put(datum.word, datum.label);
		}
	}
	
	private void baselineTest(List<Datum> testData){
		for (Datum datum : testData)
		{
			System.out.println(datum.word+": "+datum.label);
		}
	}

}
