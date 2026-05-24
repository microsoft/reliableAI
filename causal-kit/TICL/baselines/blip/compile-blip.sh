cd core

mvn clean install -DskipTests -q 

cd ../api

mvn clean package -DskipTests -q

cp target/api-1.0-jar-with-dependencies.jar ../blip.jar

