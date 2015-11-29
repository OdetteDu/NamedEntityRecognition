package cs224n.deep;

import java.util.ArrayList;
import java.util.List;

public class Sentence {
	private List<Datum> words;
	
	public Sentence() {
		this.words = new ArrayList<Datum>();
	}
	
	public Sentence(List<Datum> words) {
		this.words = words;
	}
	
	public void addWord(Datum word) {
		this.words.add(word);
	}
	
	public List<Datum> getWords() {
		return this.words;
	}
	
	@Override
	public String toString()
	{
		String sb = "";
		for (Datum word : words) {
			sb += word.toString() + "\n";
		}
		return sb;
	}
}
