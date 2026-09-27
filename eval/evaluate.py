import json
import time
from pathlib import Path
import requests

# Add the project to Python path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from manu_index import ManuIndex
from manu_embed.embed import ONNXEmbedder
from manu_index.src.datastore import LocalDataStore

from openai import OpenAI
from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
)
from deepeval.test_case import LLMTestCase

try:
    from .prompt import PROMPT
except ImportError:
    from prompt import PROMPT

api_key = "local"
base_url = "http://localhost:8888/v1"
model_name = "local"

embed_model_path = "bge_m3/onnx/model_fp16.onnx"
embed_tokenizer_path = "bge_m3"

class LocalDeepEvalLLM(DeepEvalBaseLLM):
    """DeepEval adapter for the local OpenAI-compatible model endpoint."""

    def __init__(self):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        super().__init__()

    def load_model(self):
        return self.client

    def get_model_name(self) -> str:
        return model_name

    def generate(self, prompt: str, schema=None) -> str:
        request = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        }
        if schema is not None:
            schema_json = schema.model_json_schema()
            request["messages"][0]["content"] += (
                "\n\nReturn only the evaluated JSON object. Do not return, repeat, "
                "or explain the schema."
            )
            request["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "strict": True,
                    "schema": schema_json,
                },
            }

        response = self.client.chat.completions.create(
            **request,
        )
        content = response.choices[0].message.content or ""
        if schema is not None:
            try:
                return schema.model_validate_json(content)
            except AttributeError:
                return schema.parse_raw(content)
        return content

    async def a_generate(self, prompt: str, schema=None) -> str:
        return self.generate(prompt, schema=schema)

class ArchitectureEvaluator:
    def __init__(self):
        self.llm_client = OpenAI(api_key=api_key, base_url=base_url)
        self.deep_eval_llm = LocalDeepEvalLLM()
        self.metrics = [
            AnswerRelevancyMetric(model=self.deep_eval_llm, threshold=0.5),
            FaithfulnessMetric(model=self.deep_eval_llm, threshold=0.5),
            ContextualPrecisionMetric(model=self.deep_eval_llm, threshold=0.5),
            ContextualRecallMetric(model=self.deep_eval_llm, threshold=0.5),
        ]
        
    def check_llm_availability(self) -> bool:
        """Check if the LLM endpoint is available."""
        try:
            response = requests.get(f"{base_url}/models", timeout=5)
            if response.status_code == 200:
                pass
            else:
                print(f"  ⚠ LLM endpoint returned status: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"  ⚠ Cannot connect to LLM endpoint: {e}")
            print(f"  ⚠ Will use mock responses for demonstration")
    
    def setup_index(self, sample_path: str, db_path: str) -> ManuIndex:
        """Set up ManuIndex with the new architecture."""
        try:
            # Initialize ONNX embedder
            embedder = ONNXEmbedder(
                model=embed_model_path,
                tokenizer=embed_tokenizer_path,
                max_length=1024,
                device="cuda"
            )
            
            # Initialize datastore
            datastore = LocalDataStore(db_path)
            
            # Initialize ManuIndex with local LLM config
            index = ManuIndex(
                api_key=api_key,
                model_name=model_name,
                base_url=base_url,
                embeddings=embedder,
                datastore=datastore,
                image_analyzer=None
            )
            
            # Index the sample document
            print(f"  Indexing sample document...")
            document = Path(sample_path).read_text(encoding="utf-8")
            doc_id = index.add_document(document)
            print(f"  ✓ Document indexed: {doc_id}")
            
            return index
            
        except Exception as e:
            print(f"  ✗ Index setup failed: {e}")
            raise
    
    def run_evaluation(self):
        """Run the full evaluation."""
        sample_doc = self.load_sample_document()
        test_cases = self.load_test_cases()
        
        # Check LLM availability
        self.check_llm_availability()
        
        # Setup index
        project_root = Path(__file__).parent.parent
        sample_path = project_root / "eval" / "data" / "sample.md"
        db_path = project_root / "eval" / "manu_index.db"
        
        try:
            index = self.setup_index(str(sample_path), str(db_path))
            print(f"✓ Index setup successful")
        except Exception as e:
            raise RuntimeError(f"Index setup failed: {e}")
        # Run evaluation
        return self.evaluate_with_real_results(index, sample_doc, test_cases)
    
    def load_sample_document(self) -> str:
        """Load the sample document."""
        project_root = Path(__file__).parent.parent
        sample_path = project_root / "eval" / "data" / "sample.md"
        
        with open(sample_path, "r") as f:
            content = f.read()
        
        return content
    
    def load_test_cases(self) -> list[dict]:
        """Load test cases from eval_cases.json."""
        project_root = Path(__file__).parent.parent
        test_cases_path = project_root / "eval" / "eval_cases.json"
        
        with open(test_cases_path, "r") as f:
            data = json.load(f)
        
        return data["rag_evaluation"]
    
    def evaluate_with_real_results(self, index: ManuIndex, sample_doc: str, 
                                   test_cases: list[dict]):
        """Run evaluation with simulated LLM responses."""
        print(f"\n{'='*80}")
        print(f"RUNNING EVALUATION")
        print(f"{'='*80}")
        
        total_time = 0
        successful_evaluations = 0
        
        evaluation_results = []
        
        for i, test_case_data in enumerate(test_cases, 1):
            print(f"\nTest Case {i}/{len(test_cases)}")
            
            query = test_case_data["question"]
            expected_answer = test_case_data["expected_answer"]
            
            print(f"  Query: {query[:60]}...")
            
            try:
                # Record start time
                start_time = time.time()
                
                result = index.search(query, top_k=3, top_c=3)
                
                end_time = time.time()
                query_time = end_time - start_time
                total_time += query_time

                result_str = "\n".join([f"- {chunk}" for chunk in result])
                
                response = self.llm_client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": PROMPT.format(
                                query=query,
                                context=result_str,
                            ),
                        }
                    ],
                    temperature=0,
                )
                llm_answer = response.choices[0].message.content or ""

                deep_eval_case = LLMTestCase(
                    input=query,
                    actual_output=llm_answer,
                    expected_output=expected_answer,
                    retrieval_context=result,
                )
                metric_results = {}
                for metric in self.metrics:
                    metric.measure(deep_eval_case)
                    metric_results[metric.__class__.__name__] = {
                        "score": metric.score,
                        "success": metric.success,
                        "reason": metric.reason,
                    }
                    print(
                        f"  {metric.__class__.__name__}: "
                        f"{metric.score:.3f} ({'pass' if metric.success else 'fail'})"
                    )
                
                evaluation_results.append({
                    'case': i,
                    'query': query,
                    'execution_time': query_time,
                    'retrieved_results': result,
                    'llm_answer': llm_answer,
                    'expected_answer': expected_answer,
                    'deepeval_metrics': metric_results,
                })
                
                successful_evaluations += 1
                
            except Exception as e:
                print(f"  ✗ Error during evaluation: {e}")
                evaluation_results.append({
                    'case': i,
                    'query': query,
                    'error': f"Execution: {e}"
                })
        
        return evaluation_results, total_time, successful_evaluations

    def save_evaluation_results(self, results: list[dict], total_time: float,
                                successful: int) -> Path:
        """Save evaluation parameters, summary, and per-question results as JSON."""
        output_path = Path(__file__).parent / "evaluation_result.json"
        json_results = []
        metric_scores: dict[str, list[float]] = {}

        for result in results:
            json_result = {
                "case": result.get("case"),
                "question": result.get("query"),
                "retrieve_chunks": result.get("retrieved_results"),
                "expected_answer": result.get("expected_answer"),
                "generated_answer": result.get("llm_answer"),
                "execution_time": result.get("execution_time"),
            }
            if "error" in result:
                json_result["error"] = result["error"]
            if "deepeval_metrics" in result:
                json_result["deepeval_metrics"] = result["deepeval_metrics"]
                for metric_name, metric_result in result["deepeval_metrics"].items():
                    metric_scores.setdefault(metric_name, []).append(
                        metric_result["score"]
                    )
            json_results.append(json_result)

        average_metrics = {
            metric_name: sum(scores) / len(scores)
            for metric_name, scores in metric_scores.items()
            if scores
        }
        average_time_per_query = total_time / max(len(results), 1)
        report = {
            "summary": {
                "total_evaluation": len(results),
                "successful_evaluations": successful,
                "total_execution_time": total_time,
                "avg_time_per_query": average_time_per_query,
                "average_time_per_query": average_time_per_query,
                "deepeval_average_scores": average_metrics,
            },
            "results": json_results,
        }

        with output_path.open("w", encoding="utf-8") as output_file:
            json.dump(report, output_file, indent=2, ensure_ascii=False)
            output_file.write("\n")

        print(f"\nEvaluation results saved to: {output_path}")
        return output_path

def main():
    evaluator = ArchitectureEvaluator()
    
    try:
        results, total_time, successful = evaluator.run_evaluation()
        evaluator.save_evaluation_results(results, total_time, successful)
        
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        print(f"\nFalling back to demonstration mode...")

if __name__ == "__main__":
    main()