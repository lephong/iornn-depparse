require 'depparser_rerank'
require 'utils'
require 'dict'
require 'xlua'
require 'dpiornn_gen'
require 'dp_spec'

torch.setnumthreads(1)

if #arg >= 3 then
	treebank_path = arg[2]
	kbesttreebank_path = arg[3]
	output = arg[4]

	print('load net')
	local net = IORNN:load(arg[1])
	print(net.Wh:size())

	print('create parser')
	local parser = Depparser:new(net.voca_dic, net.pos_dic, net.deprel_dic)

--	local u = arg[1]:find('/model')
--	if u == nil then parser.mail_subject = path
--	else parser.mail_subject = arg[1]:sub(1,u-1) end

	--print(parser.mail_subject)

	print('eval')
--[[
	print('\n\n--- oracle-best ---')
	parser:eval('best', kbesttreebank_path, treebank_path, output..'.oracle-best')

	print('\n\n--- oracle-worst ---')
	parser:eval('worst', kbesttreebank_path, treebank_path, output..'.oracle-worst')

	print('\n\n--- first ---')
	parser:eval('first', kbesttreebank_path, treebank_path, output..'.first')
]]
	print('\n\n--- rescore ---')
	parser:eval(net, kbesttreebank_path, treebank_path, kbesttreebank_path..'.iornnscores')

	print('\n\n--- mix. reranking ---')
	parser:eval(kbesttreebank_path..'.iornnscores', kbesttreebank_path, treebank_path, output..'.reranked')

else
	print("[network] [gold/input] [kbest] [output]")
end
