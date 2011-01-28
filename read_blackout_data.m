% read_blackout_data.m is written by Paul Hines, University of Vermont.
% (c) Paul Hines, 2009. When using this file please cite the paper:
% P. Hines, J. Apt, and S. Talukdar. Large Blackouts in North America: Historical Trends and Policy 
% Implications. Energy Policy, v. 37, pp. 5249-5259, 2009.

clear all;

% read the data
rawdata = csvread('raw_blackout_data2.csv',1);
n_records = size(rawdata,1)

% columns
cause_cols = 8:22;
region_cols = 23:29;
issame_col = 30;
pjm_col    = 31;
cols = 30;

% column names
cause_names = {'Earthquake','Tornado','Hurricane or Tropical Storm','Ice storm','Lightning',...
    'Wind/rain','Other cold weather','Fire','Intentional attack','Supply shortage','Other external cause',...
    'Equipment failure','Operator error','Voltage reduction','Voluntary reduction'};
region_names = {'West','Northeast','Midwest','Southeast','ERCOT','Hawaii','Puerto Rico'};

% collect initial columns
year     = rawdata(:,1);
month    = rawdata(:,2);
day      = rawdata(:,3);
time     = rawdata(:,4);
duration = rawdata(:,5);
MW       = rawdata(:,6);
cust     = rawdata(:,7);
causes   = rawdata(:, cause_cols);
regions  = rawdata(:,region_cols);
issame   = rawdata(:,issame_col);
%pjm     = rawdata(:,pjm_col);

% merge records
bodata = [];
i = 1;
ri = 0;
for i = 1:size(rawdata,1)
    record = rawdata(i,:);
    is_same = record(issame_col);
    if is_same
        pair = [bodata(ri,:); record];
        % combine record and prev_record
        date     = record(1:3);
        time     = min(pair(:,4));
        duration = max(pair(:,5));
        MW_cust  = sum(pair(:,6:7),1);
        cause    = pair(1,cause_cols) | pair(2,cause_cols);
        regions  = pair(1,region_cols)| pair(2,region_cols);
        %pjm      = bodata(i,pjm_col);
        record = [date time duration MW_cust cause regions 0 ];
        % place record at the end of bodata
        bodata(ri,1:cols) = record;
    else
        ri = ri + 1; % advance to the next record
        % just append the record to the data
        bodata = [bodata; record];
    end
end

n_merged = size(bodata,1)
n = n_merged;

% collect columns
year     = bodata(:,1);
month    = bodata(:,2);
day      = bodata(:,3);
time     = bodata(:,4);
duration = bodata(:,5);
MW       = bodata(:,6);
cust     = bodata(:,7);
causes   = bodata(:,cause_cols);
VR = any(causes(:,(end-1:end)),2);
regions  = bodata(:,region_cols);
same_as_prev = bodata(:,issame_col);
%pjm      = bodata(:,pjm_col);

% sort out cause categories
is_extreme_natural = any(causes(:,1:4),2);

% collect region data
West      = regions(:,1);
Northeast = regions(:,2);
Midwest   = regions(:,3);
Southeast = regions(:,4);
Texas     = regions(:,5);

% population and demand
pddata     = csvread('population_and_demand.csv',1);
pd_year    = pddata(:,1);
population = pddata(:,2);
demand     = pddata(:,3);

% customers per MW calculation
total_cust_in_2006 = 140403965;
total_MWh_in_2006 = 4093849879;
cust_per_MW = total_cust_in_2006 / (total_MWh_in_2006/8760);

% fill in missing values
subset = MW<=0 & cust>0 & ~VR;
MW_filled = MW;
MW_filled( subset ) = cust( subset ) / cust_per_MW;
subset = MW>0 & cust<=0;
cust_filled = cust .* (~VR);
cust_filled( subset ) = MW( subset ) * cust_per_MW;

% scale the data by population and demand
pop_scale_factors = population(pd_year==2000) ./ population;
demand_scale_factors = demand(pd_year==2000) ./ demand;

MW_scaled = MW_filled;
cust_scaled = cust_filled;
for y=pd_year'
    subset = year==y;
    MW_scaled(subset)   = MW_filled(subset)*demand_scale_factors(pd_year==y);
    cust_scaled(subset) = cust_filled(subset)*pop_scale_factors(pd_year==y);
end

% collect the set of years covered by the data
year_set = unique(year);
