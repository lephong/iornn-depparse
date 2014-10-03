--require 'cutils'

function trim_string(s)
  return (s:gsub("^%s*(.-)%s*$", "%1"))
end

function split_string( str , pattern )
	local pattern = pattern or "[^\t ]+"
	local toks = {}
	for k,v in string.gmatch(str, pattern) do
		toks[#toks+1] = k
	end
	return toks
end

function get_all_filenames( dir )
	local f = io.popen("ls " .. dir)
	local files = f:read("*a")
	f:close()

	return split_string(files, "[^\n]+")
end

function safe_compute_softmax(A)
	maxA,_ = A:max(1)
	maxA = torch.repeatTensor(maxA, A:size(1), 1)
	local B = torch.repeatTensor((A - maxA):exp():sum(1):log(), A:size(1), 1)
	return (A - maxA - B):exp()
end

function log_sum_of_exp(xs) 
	max,_ = xs:max(1)
	max = max[1]
	local sum = xs:add(-max):exp():sum()
	return max + math.log(sum)
end

function get_current_time()
	time = os.date("*t")
	return time.day.."/"..time.month.."/"..time.year.." "..time.hour .. ":" .. time.min .. ":" .. time.sec
end
