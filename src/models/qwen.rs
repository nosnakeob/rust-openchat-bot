extern crate intel_mkl_src;

use crate::utils::format_size;
use std::io::Write;
use std::io::{self, BufRead};
use std::path::PathBuf;
use std::process::Command;
use tokenizers::Tokenizer;

use hf_hub::api::tokio::Api;

use anyhow::{Error, Result};
use candle::quantized::gguf_file;
use candle::Tensor;
use candle_transformers::generation::{LogitsProcessor, Sampling};

use candle_examples::token_output_stream::TokenOutputStream;
use candle_transformers::models::quantized_qwen2::ModelWeights;
use candle_transformers::utils::apply_repeat_penalty;
use hf_hub::{Cache, Repo};

#[allow(unused)]
#[derive(Clone, Debug, Copy, PartialEq, Eq)]
enum Which {
    // gpu推理报错
    W2_0_5b,
    W2_1_5b,
    W2_7b,
    W2_72b,
    W25_0_5b,
    W25_1_5b,
    W25_7b,
    W25_14b,
    W25_32b,
}

impl Which {
    // 返回tokenizer, model repo, filename
    fn info(self) -> (&'static str, &'static str, &'static str, &'static str) {
        let tokenizer_fname = "tokenizer.json";
        match self {
            Which::W2_0_5b => (
                "Qwen/Qwen2-0.5B-Instruct",
                tokenizer_fname,
                "Qwen/Qwen2-0.5B-Instruct-GGUF",
                "qwen2-0_5b-instruct-q4_0.gguf",
            ),
            Which::W2_1_5b => (
                "Qwen/Qwen2-1.5B-Instruct",
                tokenizer_fname,
                "Qwen/Qwen2-1.5B-Instruct-GGUF",
                "qwen2-1_5b-instruct-q4_0.gguf",
            ),
            Which::W2_7b => (
                "Qwen/Qwen2-7B-Instruct",
                tokenizer_fname,
                "Qwen/Qwen2-7B-Instruct-GGUF",
                "qwen2-7b-instruct-q4_0.gguf",
            ),
            Which::W2_72b => (
                "Qwen/Qwen2-72B-Instruct",
                tokenizer_fname,
                "Qwen/Qwen2-72B-Instruct-GGUF",
                "qwen2-72b-instruct-q4_0.gguf",
            ),
            Which::W25_0_5b => (
                "Qwen/Qwen2.5-0.5B-Instruct",
                tokenizer_fname,
                "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
                "qwen2.5-0.5b-instruct-q4_0.gguf",
            ),
            Which::W25_1_5b => (
                "Qwen/Qwen2.5-1.5B-Instruct",
                tokenizer_fname,
                "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
                "qwen2.5-1.5b-instruct-q4_0.gguf",
            ),
            Which::W25_7b => (
                "Qwen/Qwen2.5-7B-Instruct",
                tokenizer_fname,
                "Qwen/Qwen2.5-7B-Instruct-GGUF",
                "qwen2.5-7b-instruct-q4_0.gguf",
            ),
            Which::W25_14b => (
                "Qwen/Qwen2.5-14B-Instruct",
                tokenizer_fname,
                "Qwen/Qwen2.5-14B-Instruct-GGUF",
                "qwen2.5-14b-instruct-q4_0.gguf",
            ),
            Which::W25_32b => (
                "Qwen/Qwen2.5-32B-Instruct",
                tokenizer_fname,
                "Qwen/Qwen2.5-32B-Instruct-GGUF",
                "qwen2.5-32b-instruct-q4_0.gguf",
            ),
        }
    }
}

#[derive(Debug)]
struct Args {
    /// The length of the sample to generate (in tokens).
    sample_len: usize,

    /// The temperature used to generate samples, use 0 for greedy sampling.
    temperature: f64,

    /// Nucleus sampling probability cutoff.
    top_p: Option<f64>,

    /// Only sample among the top K samples.
    top_k: Option<usize>,

    /// The seed to use when generating random samples.
    seed: u64,

    /// Run on CPU rather than GPU even if a GPU is available.
    cpu: bool,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    repeat_penalty: f32,

    /// The context size to consider for the repeat penalty.
    repeat_last_n: usize,

    /// The model size to use.
    which: Which,
}

impl Args {
    async fn tokenizer(&self) -> Result<Tokenizer> {
        let (repo, filename, _, _) = self.which.info();
        let tokenizer_path = Api::new()?.model(repo.to_string()).get(filename).await?;
        Tokenizer::from_file(tokenizer_path).map_err(Error::msg)
    }

    async fn model(&self) -> Result<PathBuf> {
        let (_, _, repo, filename) = self.which.info();

        let model_path = if let Some(path) = Cache::default()
            .repo(Repo::model(repo.to_string()))
            .get(filename)
        {
            path
        } else {
            let repo = Api::new()?.model(repo.to_string());

            let split_filenames: Vec<String> = repo
                .info()
                .await?
                .siblings
                .into_iter()
                .map(|sibling| sibling.rfilename)
                .filter(|s| s.starts_with(filename.strip_suffix(".gguf").unwrap()))
                .collect();

            let mut split_paths = vec![];
            for filename in &split_filenames {
                split_paths.push(repo.get(filename).await?);
            }

            // 获取输出目录
            let output_dir = split_paths[0].parent().unwrap();
            // println!("{:?}", output_dir);

            let merged_path = output_dir.join(filename);

            // 构建命令
            let exe_path = which::which("llama-gguf-split")?;
            let mut command = Command::new(exe_path);
            command
                .arg("--merge")
                .arg(split_paths[0].to_str().unwrap())
                .arg(merged_path.to_str().unwrap());

            // println!("{:?}", command);

            // 执行命令
            let output = command.output()?;

            if !output.status.success() {
                let error = String::from_utf8_lossy(&output.stderr);
                return Err(Error::msg(format!("gguf-utils merge failed: {}", error)));
            }

            let merged_path = output_dir.join(filename);
            merged_path
        };

        Ok(model_path)
    }
}

impl Default for Args {
    fn default() -> Self {
        Self {
            sample_len: 1000,
            temperature: 0.8,
            top_p: None,
            top_k: None,
            seed: 299792458,
            cpu: false,
            repeat_penalty: 1.1,
            repeat_last_n: 64,
            which: Which::W25_7b,
        }
    }
}

async fn setup_model(args: &Args) -> Result<(ModelWeights, candle::Device)> {
    let device = candle_examples::device(args.cpu)?;
    let model_path = args.model().await?;
    let mut file = std::fs::File::open(&model_path)?;
    let start = std::time::Instant::now();

    // 构建模型
    let model = {
        let model = gguf_file::Content::read(&mut file).map_err(|e| e.with_path(&model_path))?;
        let mut total_size_in_bytes = 0;
        for (_, tensor) in model.tensor_infos.iter() {
            let elem_count = tensor.shape.elem_count();
            total_size_in_bytes +=
                elem_count * tensor.ggml_dtype.type_size() / tensor.ggml_dtype.block_size();
        }
        println!(
            "loaded {:?} tensors ({}) in {:.2}s",
            model.tensor_infos.len(),
            &format_size(total_size_in_bytes),
            start.elapsed().as_secs_f32(),
        );
        ModelWeights::from_gguf(model, &mut file, &device)?
    };

    Ok((model, device))
}

async fn setup_tokenizer(args: &Args) -> Result<TokenOutputStream> {
    let tokenizer = args.tokenizer().await?;
    Ok(TokenOutputStream::new(tokenizer))
}

fn setup_logits_processor(args: &Args) -> LogitsProcessor {
    let temperature = args.temperature;
    let sampling = if temperature <= 0. {
        Sampling::ArgMax
    } else {
        match (args.top_k, args.top_p) {
            (None, None) => Sampling::All { temperature },
            (Some(k), None) => Sampling::TopK { k, temperature },
            (None, Some(p)) => Sampling::TopP { p, temperature },
            (Some(k), Some(p)) => Sampling::TopKThenTopP { k, p, temperature },
        }
    };
    LogitsProcessor::from_sampling(args.seed, sampling)
}

fn fmt_prompt(prompt: &str) -> String {
    format!(
        "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
        prompt
    )
}

fn get_user_prompt() -> String {
    println!("请输入您的问题:");
    let stdin = io::stdin();
    let mut line = String::new();
    stdin
        .lock()
        .read_line(&mut line)
        .expect("Failed to read line");

    // 去除末尾的换行符
    line = line.trim().to_string();

    line
}

fn token2str(token: u32, tokenizer: &Tokenizer) -> Result<String> {
    tokenizer.decode(&[token], true).map_err(Error::msg)
}

// 处理提示词
fn process_prompt(prompt: &str, tokenizer: &Tokenizer) -> Result<Vec<u32>> {
    // 格式化提示词
    let prompt = fmt_prompt(&prompt);

    // 将提示词转换为token
    let tokens = tokenizer.encode(prompt, true).map_err(Error::msg)?;
    let tokens = tokens.get_ids().to_vec();

    Ok(tokens)
}

fn gen_next_token(
    ctx_tokens: &[u32],
    idx_pos: usize,
    model: &mut ModelWeights,
    device: &candle::Device,
    logits_processor: &mut LogitsProcessor,
    ans_start_idx: Option<usize>,
    repeat_penalty: Option<f32>,
    repeat_last_n: Option<usize>,
) -> Result<u32> {
    let logits = if let (Some(ans_start_idx), Some(repeat_penalty), Some(repeat_last_n)) =
        (ans_start_idx, repeat_penalty, repeat_last_n)
    {
        let input = Tensor::new(&[*ctx_tokens.last().unwrap()], device)?.unsqueeze(0)?;
        let mut logits = model.forward(&input, idx_pos)?;
        logits = logits.squeeze(0)?;

        if repeat_penalty != 1. {
            let ans_tokens = &ctx_tokens[ans_start_idx..];
            let start_at = ans_tokens.len().saturating_sub(repeat_last_n);
            logits = apply_repeat_penalty(&logits, repeat_penalty, &ans_tokens[start_at..])?
        };

        logits
    } else {
        let input = Tensor::new(ctx_tokens, device)?.unsqueeze(0)?;
        let logits = model.forward(&input, idx_pos)?;
        logits.squeeze(0)?
    };

    logits_processor.sample(&logits).map_err(Error::msg)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::Tensor;
    use candle_transformers::utils::apply_repeat_penalty;
    use std::env;
    use tokio::select;

    // 用tos输出奇怪
    #[tokio::test]
    async fn test_prompt() -> Result<()> {
        env::set_var("HTTPS_PROXY", "http://127.0.0.1:10808");
        let args = Args::default();
        println!("{args:?}");

        // 初始化模型、分词器和logits处理器
        let (mut model, device) = setup_model(&args).await?;
        let mut tos = setup_tokenizer(&args).await?;
        let mut logits_processor = setup_logits_processor(&args);

        // 初始化上下文token列表
        let mut ctx_tokens = vec![];
        let eos_token = *tos.tokenizer().get_vocab(true).get("<|im_end|>").unwrap();
        let to_sample = args.sample_len.saturating_sub(1);

        let prompts = vec![
            "我是snake，你给我记住了",
            "还记得我是谁吗",
            "你是谁",
            "给我笑一笑",
        ];

        for prompt_str in prompts {
            let prompt_tokens = process_prompt(&prompt_str, tos.tokenizer())?;
            ctx_tokens.extend_from_slice(&prompt_tokens);

            let start = std::time::Instant::now();

            // 生成第一个token
            let mut next_token = gen_next_token(
                &ctx_tokens,
                0,
                &mut model,
                &device,
                &mut logits_processor,
                None,
                None,
                None,
            )?;
            let ans_start_idx = ctx_tokens.len();
            ctx_tokens.push(next_token);
            if let Some(t) = tos.next_token(next_token)? {
                print!("{t}");
                io::stdout().flush()?;
            }

            // 循环生成回答
            for index in 0..to_sample {
                next_token = gen_next_token(
                    &ctx_tokens,
                    ans_start_idx + index,
                    &mut model,
                    &device,
                    &mut logits_processor,
                    Some(ans_start_idx),
                    Some(args.repeat_penalty),
                    Some(args.repeat_last_n),
                )?;
                ctx_tokens.push(next_token);

                if let Some(t) = tos.next_token(next_token)? {
                    print!("{t}");
                    io::stdout().flush()?;
                }

                if next_token == eos_token {
                    break;
                }
            }

            let dt = start.elapsed();

            println!(
                "\n\n生成速度: {:.2} token/s",
                (ctx_tokens.len() - ans_start_idx) as f64 / dt.as_secs_f64(),
            );
        }

        env::remove_var("HTTPS_PROXY");
        Ok(())
    }

    #[tokio::test]
    async fn test_prompt_wo_tos() -> Result<()> {
        env::set_var("HTTPS_PROXY", "http://127.0.0.1:10808");
        let args = Args::default();
        println!("{args:?}");

        // 初始化模型、分词器和logits处理器
        let (mut model, device) = setup_model(&args).await?;
        let tokenizer = args.tokenizer().await?;
        let mut logits_processor = setup_logits_processor(&args);

        // 初始化上下文token列表
        let mut ctx_tokens = vec![];
        let eos_token = *tokenizer.get_vocab(true).get("<|im_end|>").unwrap();
        let to_sample = args.sample_len.saturating_sub(1);

        let prompts = vec![
            "我是snake，你给我记住了",
            "还记得我是谁吗",
            "你是谁",
            "给我笑一笑",
        ];

        for prompt_str in prompts {
            // 将提示词转换为token并添加到上下文
            let prompt_tokens = process_prompt(&prompt_str, &tokenizer)?;
            ctx_tokens.extend_from_slice(&prompt_tokens);

            let start = std::time::Instant::now();

            // 生成第一个token
            let mut next_token = gen_next_token(
                &ctx_tokens,
                0,
                &mut model,
                &device,
                &mut logits_processor,
                None,
                None,
                None,
            )?;
            let ans_start_idx = ctx_tokens.len();
            ctx_tokens.push(next_token);
            if let Ok(t) = token2str(next_token, &tokenizer) {
                print!("{t}");
                io::stdout().flush()?;
            }

            // 循环生成回答
            for index in 0..to_sample {
                next_token = gen_next_token(
                    &ctx_tokens,
                    ans_start_idx + index,
                    &mut model,
                    &device,
                    &mut logits_processor,
                    Some(ans_start_idx),
                    Some(args.repeat_penalty),
                    Some(args.repeat_last_n),
                )?;
                ctx_tokens.push(next_token);

                if let Ok(t) = token2str(next_token, &tokenizer) {
                    print!("{t}");
                    io::stdout().flush()?;
                }

                if next_token == eos_token {
                    break;
                }
            }

            // 将回答添加到上下文
            let dt = start.elapsed();

            println!(
                "\n\n生成速度: {:.2} token/s",
                (ctx_tokens.len() - ans_start_idx) as f64 / dt.as_secs_f64(),
            );
        }

        env::remove_var("HTTPS_PROXY");
        Ok(())
    }

    #[tokio::test]
    async fn test_gen() -> Result<()> {
        env::set_var("HTTPS_PROXY", "http://127.0.0.1:10808");
        let args = Args::default();
        println!("{args:?}");

        let callback = |t| {
            print!("{t}");
            io::stdout().flush().unwrap();
        };

        // 初始化模型、分词器和logits处理器
        let (mut model, device) = setup_model(&args).await?;
        let tokenizer = args.tokenizer().await?;
        let mut logits_processor = setup_logits_processor(&args);

        // 初始化上下文token列表
        let mut ctx_tokens = vec![];
        let eos_token = *tokenizer.get_vocab(true).get("<|im_end|>").unwrap();
        let to_sample = args.sample_len.saturating_sub(1);

        loop {
            // 获取用户输入
            let prompt_str = get_user_prompt();

            let prompt_tokens = process_prompt(&prompt_str, &tokenizer)?;
            ctx_tokens.extend_from_slice(&prompt_tokens);

            let start = std::time::Instant::now();

            // 生成第一个token
            let mut next_token = gen_next_token(
                &ctx_tokens,
                0,
                &mut model,
                &device,
                &mut logits_processor,
                None,
                None,
                None,
            )?;
            let ans_start_idx = ctx_tokens.len();
            ctx_tokens.push(next_token);
            token2str(next_token, &tokenizer).map(callback)?;

            // 循环生成回答
            for index in 0..to_sample {
                next_token = gen_next_token(
                    &ctx_tokens,
                    ans_start_idx + index,
                    &mut model,
                    &device,
                    &mut logits_processor,
                    Some(ans_start_idx),
                    Some(args.repeat_penalty),
                    Some(args.repeat_last_n),
                )?;
                ctx_tokens.push(next_token);

                token2str(next_token, &tokenizer).map(callback)?;

                if next_token == eos_token {
                    break;
                }
            }

            let dt = start.elapsed();
            println!(
                "\n\n生成速度: {:.2} token/s",
                (ctx_tokens.len() - ans_start_idx) as f64 / dt.as_secs_f64(),
            );
        }

        env::remove_var("HTTPS_PROXY");
        Ok(())
    }
}
