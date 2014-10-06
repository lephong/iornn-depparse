require 'depstruct'
require 'utils'
require 'dict'
require 'xlua'
require 'dp_spec'

p = xlua.Profiler()

Depparser = {}
Depparser_mt = { __index = Depparser }

function Depparser:new(voca_dic, pos_dic, deprel_dic)
	local parser = { voca_dic = voca_dic, pos_dic = pos_dic, deprel_dic = deprel_dic}
	parser.punc_list = {
		--[voca_dic.word2id['=']] = 1,
		[voca_dic.word2id['-']] = 1,
		[voca_dic.word2id['--']] = 1,
		[voca_dic.word2id[',']] = 1,
		[voca_dic.word2id[';']] = 1,
		[voca_dic.word2id[':']] = 1,
		[voca_dic.word2id['!']] = 1,
		[voca_dic.word2id['?']] = 1,
		--[voca_dic.word2id['/']] = 1,
		[voca_dic.word2id['.']] = 1,
		[voca_dic.word2id['...']] = 1,
		[voca_dic.word2id["'"]] = 1,
		[voca_dic.word2id["''"]] = 1,
		[voca_dic.word2id['(']] = 1,
		[voca_dic.word2id[')']] = 1,
		[voca_dic.word2id['{']] = 1,
		[voca_dic.word2id['}']] = 1,
		--[voca_dic.word2id['@']] = 1,
		[voca_dic.word2id['*']] = 1,
		--[voca_dic.word2id['\\*']] = 1,
		--[voca_dic.word2id['\\*\\*']] = 1,
		[voca_dic.word2id['&']] = 1,
		[voca_dic.word2id['#']] = 1,
		[voca_dic.word2id['%']] = 1,
	}
	
	setmetatable(parser, Depparser_mt)
	return parser
end

function Depparser:load_dsbank(path)
	local dsbank = {}
	local raw = {}

	tokens = {}
	local i = 1
	for line in io.lines(path) do
		--print(line)
		line = trim_string(line)
		if line == '' then
			local status, err =  pcall( function() 
					ds,sent = Depstruct:create_from_strings(tokens, 
						self.voca_dic, self.pos_dic, self.deprel_dic) 
				end ) 
			if status then
				dsbank[#dsbank+1] = ds
				raw[#raw+1] = sent
			else 
				print(err)
			end
			tokens = {}

			if math.mod(i,100) == 0 then io.write(i..' trees \r'); io.flush() end
			i = i + 1
		else 
			tokens[#tokens+1] = line
		end
	end
	dsbank.raw = raw

	return dsbank, raw
end

function Depparser:load_kbestdsbank(path, golddsbank)
	local dsbank, raw = self:load_dsbank(path)
	local kbestdsbank = {}

	local group = nil
	for i,ds in ipairs(dsbank) do
		if group == nil or group[1].n_words ~= ds.n_words or (group[1].word - ds.word):abs():sum() > 0 then
			group = { ds , raw = {raw[i]} }
			kbestdsbank[#kbestdsbank+1] = group
		else
			group[#group+1] = ds
			group.raw[#group] = raw[i]
		end
	end

	for i,group in ipairs(kbestdsbank) do
		local goldds = golddsbank[i]
		for _,ds in ipairs(group) do
			if ds.n_words ~= goldds.n_words then 
				print(#group)
				print(#kbestdsbank)
				print(ds.n_words)
				print(goldds.n_words)
				error("not match")
			end
			ds.word = goldds.word:clone() 
			ds.cap = goldds.cap:clone()
		end
	end
	
	return kbestdsbank
end

function Depparser:train(net, traintrebank_path, devdsbank_path, kbestdevdsbank_path, model_dir)
	print('load train dsbank')
	local traindsbank,_ = self:load_dsbank(traintrebank_path)
	
	net.update_L = TRAIN_UPDATE_L

	-- shuf the traindsbank
	print('shufing train dsbank')
	local new_i = torch.randperm(#traindsbank)
	temp = {}
	for i = 1,#traindsbank do
		temp[i] = traindsbank[new_i[i]]
	end
	traindsbank = temp

	-- train
	local adagrad_config = {	weight_learningRate	= TRAIN_WEIGHT_LEARNING_RATE,
								voca_learningRate	= TRAIN_VOCA_LEARNING_RATE	}
	local adagrad_state = {}

	net:save(model_dir .. '/model_0')
	local prefix = model_dir..'/model'

	print('train net')
	adagrad_config, adagrad_state = net:train_with_adagrad(traindsbank, TRAIN_BATCHSIZE,
										TRAIN_MAX_N_EPOCHS, {lambda = TRAIN_LAMBDA, lambda_L = TRAIN_LAMBDA_L}, 
										prefix, adagrad_config, adagrad_state, 
										devdsbank_path, kbestdevdsbank_path)
	return net
end

function Depparser:compute_scores(test_ds, gold_ds, punc)
	local unlabel = 0
	local label = 0
	local n = 0


	if test_ds.n_words ~= gold_ds.n_words then
		error('not match')
	else
		for j = 2,test_ds.n_words do -- ignore ROOT
			local count = true
			--local w = self.voca_dic.id2word[gold_ds.word[j]]:gsub("%p",'')
			
			if punc == false and self.punc_list[gold_ds.word[j]] == 1 then --string.len(w) == 0 then
				count = false
			end
			
			if count then
				n = n + 1	 
				if test_ds.head[j] == gold_ds.head[j] then
					unlabel = unlabel + 1
					if test_ds.deprel[j] == gold_ds.deprel[j] then
						label = label + 1
					end
				end
			end
		end
	end
	return { unlabel = unlabel , label = label , n = n }
end

function Depparser:rerank_oracle(kbestdsbank, golddsbank, typ)
	local typ = typ or 'best'
	local K = 100000

	if #kbestdsbank ~= #golddsbank then 
		error('size not match')
	end

	local ret = {raw = {}}

	for i,parses in ipairs(kbestdsbank) do
		if typ == 'first' then ret[i] = parses[1]; ret.raw[i] = parses.raw[1] 
		else
			local gold = golddsbank[i]
			local best_parse = nil
			local best_score = nil
			local best_raw = nil

			for k,parse in ipairs(parses) do
				if k > K then break end
				local score = self:compute_scores(parse, gold)
				if typ == 'worst' then score.unlabel = -score.unlabel end
				if best_score == nil or score.unlabel > best_score then
					best_parse = parse
					best_score = score.unlabel
					best_raw = parses.raw[k]
				end
			end
			ret[i] = best_parse
			ret.raw[i] = best_raw
		end
	end

	return ret
end

function Depparser:rerank_scorefile(netscorefile, mstscorefile, kbestdsbank, kbestdsscore, alpha, K)
	local K = K or 10
	local ret = { raw = {} }
	local sum_sen_log_p = 0
	local sum_n_words = 0

	local f = torch.DiskFile(netscorefile, 'r')
	if f:readInt() ~= #kbestdsbank then 
		error('not match')
	end

	local i = 1
	for line in io.lines(mstscorefile) do
		local org_parses = kbestdsbank[i]
		local parses = { raw = {} }

		local best_parse = nil
		local best_score = nil
		local best_raw = nil
		local log_probs = f:readObject()

		local mstscores = split_string(line)

		for k = 1, math.min(#log_probs, math.min(K, #org_parses)) do
			parses[k] = org_parses[k]
			parses.raw[k] = org_parses.raw[k]
		end

		for j,parse in ipairs(parses) do
			local score = tonumber(mstscores[j]) * alpha + (1-alpha)*log_probs[j]
			if best_score == nil or score > best_score then
				best_parse = parse
				best_score = score
				best_raw = parses.raw[j]
			end
		end
		ret[i] = best_parse
		ret.raw[i] = best_raw

		sum_sen_log_p = sum_sen_log_p + log_sum_of_exp(torch.Tensor(log_probs))
		sum_n_words = sum_n_words + parses[1].n_words - 1

		i = i + 1
	end
	local ppl = math.pow(2, -sum_sen_log_p / math.log(2) / sum_n_words)

	f:close()
	
	return ret, ppl

end

function Depparser:rerank(net, kbestdsbank, output)
	local K = K or 10000
	local ret = {}
	local sum_sen_log_p = 0
	local sum_n_words = 0

	local f = nil
	if output then
		f = torch.DiskFile(output, 'w')
		f:writeInt(#kbestdsbank)
	end

	for i,org_parses in ipairs(kbestdsbank) do
		if math.mod(i, 10) == 0 then io.write(i..'\r'); io.flush() end
		local parses = {}
		for k = 1, math.min(K, #org_parses) do
			parses[k] = org_parses[k]
		end

		local best_parse = nil
		local best_score = nil
		local log_probs = net:compute_log_prob(parses)

		if f then f:writeObject(log_probs) end

		for j,parse in ipairs(parses) do
			if best_score == nil or log_probs[j] > best_score then
				best_parse = parse
				best_score = log_probs[j]
			end
		end
		ret[i] = best_parse
		sum_sen_log_p = sum_sen_log_p + log_sum_of_exp(torch.Tensor(log_probs))
		sum_n_words = sum_n_words + parses[1].n_words - 1
	end
	local ppl = math.pow(2, -sum_sen_log_p / math.log(2) / sum_n_words)

	if f then f:close() end
	
	return ret, ppl
end

function Depparser:perplexity(net, dsbank)
	local log_probs = net:compute_log_prob(dsbank)
	local sum = 0
	local nwords = 0
	for i,log_p in ipairs(log_probs) do
		sum = sum + log_p / math.log(2)
		nwords = nwords + dsbank[i].n_words - 1
	end
	return math.pow(2, -sum / nwords)
end

function Depparser:computeAScores(parses, golddsbank, punc, output)
	if output then
		--print('print to '..output)
		local f = io.open(output, 'w')
		for i, parse in ipairs(parses) do
			local sent = parses.raw[i]
			for j = 2,#sent do
				f:write((j-1)..'\t'..sent[j]..'\t_\t_\t_\t_\t'..(parse.head[j]-1)..'\t'..self.deprel_dic.id2word[parse.deprel[j] ]..'\t_\t_\n')
			end
			f:write('\n')	
		end
		f:close()
	end

	-- compute scores
	local total = 0
	local label = 0
	local unlabel = 0

	for i,parse in ipairs(parses) do
		local gold = golddsbank[i]
		local ret = self:compute_scores(parse, gold, punc)
		total 	= total + ret.n
		label 	= label + ret.label
		unlabel = unlabel + ret.unlabel
	end

	--print(total)

	local LAS = label / total * 100
	local UAS = unlabel / total * 100
	return LAS, UAS
end

-- should not call it directly when training, there's a mem-leak problem!!!
function Depparser:eval(typ, kbestpath, goldpath, output)
	local str = ''

	print('load ' .. goldpath)
	local golddsbank, raw = self:load_dsbank(goldpath)

	print('load ' .. kbestpath)
	local kbestdsbank, kbestdsscore  = self:load_kbestdsbank(kbestpath, golddsbank)

	print('reranking...')

	-- compute perplexity
	if type(typ) ~= 'string' then
		--str = str .. 'ds-ppl ' .. self:perplexity(typ, golddsbank) .. '\n'
	end

	-- reranking
	local parses = nil
	local ppl = nil
	if type(typ) == 'string' then
		if typ == 'best' or typ == 'worst' or typ == 'first' then 
			parses = self:rerank_oracle(kbestdsbank, golddsbank, typ)
			LAS, UAS = self:computeAScores(parses, golddsbank, punc, output) --'/tmp/univ-test-result.conll.'..typ)
			str = 'LAS = ' .. string.format("%.2f",LAS)..'\nUAS = ' ..string.format("%.2f",UAS)
			print(str)
		else 
			self.mail_subject = nil

			if K_range == nil then K_range = {K,K} end
			if alpha_range == nil then alpha_range = {alpha,alpha} end
			print('k\talpha\tUAS\tLAS')

			-- search for best K and alpha
			for k = K_range[1],K_range[2] do
				best_alpha = 0
				best_UAS = 0
				best_LAS = 0
				for a = alpha_range[1],alpha_range[2],0.005 do
					parses = self:rerank_scorefile(typ, kbestpath..'.mstscores', kbestdsbank, kbestdsscore, a, k)
					LAS, UAS = self:computeAScores(parses, golddsbank, punc, output) 
					if UAS > best_UAS then 
						best_UAS = UAS
						best_alpha = a
						best_LAS = LAS
					end
				end
				print(k .. '\t' .. best_alpha .. '\t' .. string.format('%.2f',best_UAS) .. '\t' .. string.format('%.2f',best_LAS))
			end
		end
	else 
		local net = typ
		parses,ppl = self:rerank(net, kbestdsbank, kbestpath..'.iornnscores')
		LAS, UAS = self:computeAScores(parses, golddsbank, punc)
		str = 'LAS = ' .. string.format("%.2f",LAS)..'\nUAS = ' ..string.format("%.2f",UAS)
		print(str)

		--[[ mail
		if EVAL_EMAIL_ADDR and self.mail_subject then
			os.execute('echo "'..str..'" | mail -s '..self.mail_subject..' '..EVAL_EMAIL_ADDR)
		end 
		print('sen-ppl ' .. ppl ..'\n')
		]]
	end	
end

--[[ for testing
torch.setnumthreads(1)

print('load dictionaries')
local data_path = '../data/wsj-dep/universal/'

local voca_dic = Dict:new()
voca_dic:load(data_path .. '/dic/collobert/words.lst')
local pos_dic = Dict:new()
pos_dic:load(data_path .. '/dic/cpos.lst')
local deprel_dic = Dict:new()
deprel_dic:load(data_path .. '/dic/deprel.lst')

local parser = Depparser:new(voca_dic, pos_dic, deprel_dic)

print('eval...')
parser:eval(nil, data_path..'/data/dev-small-10best-mst.conll', data_path..'/data/dev-small.conll')
]]
