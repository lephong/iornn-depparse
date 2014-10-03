require 'dp_spec'

Dict = {}
Dict_mt = {__index=Dict}

--***************** construction **************
function Dict:new(wfunc)
	local d = {}
	d.word2id = {} --['UNKNOWN'] = 1}
	d.id2word = {} --'UNKNOWN'}
	d.func = wfunc
	d.size = 0

	setmetatable(d, Dict_mt)

	return d
end

function Dict:setmetatable(dic)
	setmetatable(dic, Dict_mt)
	return dic
end

--**************** load from file ************
function Dict:load(filename)
	local i = 1
	for line in io.lines(filename) do
		--self:addword(line)
		word = line
		local id = self.word2id[word]
		if id == nil then
			id = self.size + 1
			self.word2id[word] = id
			self.id2word[id] = word
			self.size = self.size + 1
		end
	end
end

--***************** size ****************
-- do not use it
function Dict:size()
	return #self.id2word
end


function Dict:addword(word)
	if self.func ~= nil then 
		word = self.func(word)
	end

	local id = self.word2id[word]
	if id == nil then
		id = self.size + 1
		self.word2id[word] = id
		self.id2word[id] = word
		self.size = self.size + 1
	end
	return id
end

function Dict:get_id(word)
	if self.func ~= nil then 
		word = self.func(word)
	end
	local ret = self.word2id[word]
	if ret == nil then 
		ret = self.word2id['UNKNOWN']
	end
	return ret
end

--***************** template function *************--
function huang_template(word)

		if word == 'UNKNOWN' or word == 'PADDING' then 
			return word
		elseif word == '-LRB-' then return '('
		elseif word == '-RRB-' then return ')'
		elseif word == '-LSB-' then return '['
		elseif word == '-RSB-' then return ']'
		elseif word == '-LCB-' then return '{'
		elseif word == '-RCB-' then return '}'
		else
			return string.lower(string.gsub(word, '[0-9]', 'CD'))
		end
end

function collobert_template(word)
	if word == 'UNKNOWN' or word == 'PADDING' then 
		return word
	elseif word == '-LRB-' then return '('
	elseif word == '-RRB-' then return ')'
	elseif word == '-LSB-' then return '['
	elseif word == '-RSB-' then return ']'
	elseif word == '-LCB-' then return '{'
	elseif word == '-RCB-' then return '}'
	else
		return string.lower(string.gsub(word, '[0-9]', '0'))
	end
end

function mssc_template(word)
	if word == 'UNKNOWN' or word == 'PADDING' then 
		return word
	else
		return string.lower(string.gsub(word, '[0-9]', '0'))
	end
end

function blacoe_template(word)
	if word == 'UNKNOWN' or word == 'PADDING' then 
		return word
	elseif word == '-LRB-' then return '('
	elseif word == '-RRB-' then return ')'
	elseif word == '-LSB-' then return '['
	elseif word == '-RSB-' then return ']'
	elseif word == '-LCB-' then return '{'
	elseif word == '-RCB-' then return '}'
	else
		return string.lower(word)
	end
end

function turian_template(word)
	if word == 'UNKNOWN' then 
		return word
	elseif word == '-LRB-' then return '('
	elseif word == '-RRB-' then return ')'
	elseif word == '-LSB-' then return '['
	elseif word == '-RSB-' then return ']'
	elseif word == '-LCB-' then return '{'
	elseif word == '-RCB-' then return '}'
	else
		return word
	end
end

function Dict:get_cap_feature(str)
	local str_l = str:lower()
	local str_u = str:upper()

	if 		str == str_l then return ALL_LOWER
	elseif 	str == str_u then return ALL_UPPER
	elseif	str:sub(1,1) == str_u:sub(1,1) then return FIRST_UPPER
	else return NOT_FIRST_UPPER
	end
end
