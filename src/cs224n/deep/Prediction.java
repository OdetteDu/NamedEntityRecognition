package cs224n.deep;

public class Prediction {

	private final String word;
	private final String goldLabel;
	private String predictedLabel;
	
	public Prediction(String word, String goldLabel, String predictedLabel)
	{
		this.word = word;
		this.goldLabel = goldLabel;
		this.predictedLabel = predictedLabel;
	}
	
	public String getWord()
	{
		return this.word;
	}
	
	public String getGoldLabel()
	{
		return this.goldLabel;
	}

	public String getPredictedLabel() {
		return predictedLabel;
	}

	public void setPredictedLabel(String predictedLabel) {
		this.predictedLabel = predictedLabel;
	}
	
	@Override
	public String toString()
	{
		return this.word + "\t" + this.goldLabel + "\t" + this.predictedLabel;
	}
}
