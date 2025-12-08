function allSessions = load_all_data(dataFolder, defaultFs)

allSessions = [];
k = 0;

for userID = 1:10
    for day = 1:2
        
        if day == 1
            dayTag = 'FD';
        else
            dayTag = 'MD';
        end
        
        fileName = fullfile(dataFolder, sprintf('U%dNW_%s.csv', userID, dayTag));
        
        if ~isfile(fileName)
            warning('File not found: %s (skipping)', fileName);
            continue;
        end
        
        raw = readmatrix(fileName);
        
        if size(raw,2) < 7
            error('File %s has fewer than 7 columns.', fileName);
        end
        
        Ax = raw(:,2); Ay = raw(:,3); Az = raw(:,4);
        Gx = raw(:,5); Gy = raw(:,6); Gz = raw(:,7);
        
        sig = [Ax, Ay, Az, Gx, Gy, Gz];
        
        k = k + 1;
        allSessions(k).signal = sig;
        allSessions(k).fs     = defaultFs;
        allSessions(k).userID = userID;
        allSessions(k).day    = day;
    end
end

end
